#!/usr/bin/env python3
"""Submit LLaMA-Factory training jobs through Slurm.

The launcher creates a run directory containing:
  - a runtime copy of the training YAML
  - an optional DeepSpeed ZeRO config
  - the generated sbatch script or existing-allocation script
  - launcher and submission logs in the selected log directory
  - Slurm and per-node logs

By default it submits the generated script with sbatch unless --dry-run is used.
If --jobid is provided, it starts an srun step inside that existing allocation.
Default sbatch submission uses sbatch --wait so the launcher exits when the job ends.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = ROOT / "runs"


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._-") or "llamafactory"


def resolve_existing_path(path: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"{label} does not exist: {resolved}")
    return resolved


def make_deepspeed_config(zero_stage: int) -> dict:
    base: dict = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "zero_allow_untested_optimizer": True,
        "bf16": {"enabled": "auto"},
        "fp16": {"enabled": "auto"},
    }

    if zero_stage == 2:
        base["zero_optimization"] = {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 500_000_000,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 500_000_000,
            "contiguous_gradients": True,
        }
    elif zero_stage == 3:
        base["zero_optimization"] = {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1_000_000_000,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1_000_000_000,
            "stage3_max_reuse_distance": 1_000_000_000,
            "stage3_gather_16bit_weights_on_model_save": True,
        }
    else:
        raise ValueError(f"unsupported ZeRO stage: {zero_stage}")

    return base


def resolve_deepspeed_config_from_dir(deepspeed_dir: str, zero_stage: int, config_name: str | None) -> Path:
    directory = resolve_existing_path(deepspeed_dir, "DeepSpeed config directory")
    if not directory.is_dir():
        raise SystemExit(f"DeepSpeed config directory is not a directory: {directory}")

    if config_name:
        candidate = directory / config_name.format(stage=zero_stage)
        if not candidate.exists():
            raise SystemExit(f"DeepSpeed config does not exist: {candidate}")
        return candidate.resolve()

    names = [
        f"deepspeed_zero{zero_stage}.json",
        f"ds_zero{zero_stage}.json",
        f"zero{zero_stage}.json",
        f"ds_z{zero_stage}_config.json",
        f"z{zero_stage}.json",
    ]
    candidates = [directory / name for name in names]
    existing = [candidate.resolve() for candidate in candidates if candidate.exists()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        joined = "\n  ".join(str(path) for path in existing)
        raise SystemExit(
            "Multiple matching DeepSpeed configs found. Use --deepspeed-config-name to choose one:\n  "
            + joined
        )

    expected = "\n  ".join(str(candidate) for candidate in candidates)
    raise SystemExit(
        "Could not find a DeepSpeed config in --deepspeed-dir. Expected one of:\n  "
        + expected
        + "\nOr pass --deepspeed-config-name."
    )


def remove_top_level_yaml_key(text: str, key: str) -> str:
    """Remove a simple top-level YAML key block from a runtime copy.

    This intentionally avoids requiring PyYAML on login nodes. It is scoped to
    top-level keys like `deepspeed:` and keeps the user's source YAML unchanged.
    """

    lines = text.splitlines()
    output: list[str] = []
    skip = False
    key_pattern = re.compile(rf"^{re.escape(key)}\s*:")
    top_level_pattern = re.compile(r"^[A-Za-z0-9_\"'-][^:]*:")

    for line in lines:
        if key_pattern.match(line):
            skip = True
            continue

        if skip:
            if line and not line.startswith((" ", "\t")) and top_level_pattern.match(line):
                skip = False
                output.append(line)
            elif line and not line.startswith((" ", "\t")) and not line.lstrip().startswith("#"):
                skip = False
                output.append(line)
            else:
                continue
        else:
            output.append(line)

    return "\n".join(output).rstrip() + "\n"


def write_runtime_config(
    source_config: Path,
    runtime_config: Path,
    zero_stage: int,
    deepspeed_config: Path | None,
) -> None:
    text = source_config.read_text(encoding="utf-8")
    text = remove_top_level_yaml_key(text, "deepspeed")

    if zero_stage in (2, 3):
        if deepspeed_config is None:
            raise ValueError("deepspeed_config is required for ZeRO-2/ZeRO-3")
        text = text.rstrip() + f"\ndeepspeed: {deepspeed_config}\n"

    runtime_config.write_text(text, encoding="utf-8")


def normalize_sbatch_extra(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for raw in values:
        value = raw.strip()
        if not value:
            continue
        if value.startswith("#SBATCH"):
            normalized.append(value)
        elif value.startswith("--"):
            normalized.append(f"#SBATCH {value}")
        else:
            normalized.append(f"#SBATCH --{value}")
    return normalized


def parse_env_pairs(values: Iterable[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"environment override must be KEY=VALUE, got {value!r}")
        key, env_value = value.split("=", 1)
        key = key.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            raise SystemExit(f"invalid environment variable name: {key!r}")
        env[key] = env_value
    return env


def default_train_command(runtime_config: Path) -> str:
    return f"llamafactory-cli train {shell_quote(runtime_config)}"


def build_train_command(template: str | None, runtime_config: Path, workdir: Path) -> str:
    if not template:
        return default_train_command(runtime_config)
    return template.format(
        config=shell_quote(runtime_config),
        workdir=shell_quote(workdir),
    )


def render_export_lines(env: dict[str, str]) -> list[str]:
    return [f"export {key}={shell_quote(value)}" for key, value in sorted(env.items())]


def render_pre_commands(commands: Iterable[str]) -> list[str]:
    return [command.strip() for command in commands if command.strip()]


def render_srun_common_options(args: argparse.Namespace) -> str:
    options: list[str] = []
    if args.srun_cpu_bind and args.srun_cpu_bind != "default":
        options.append(f"--cpu-bind={args.srun_cpu_bind}")
    return " ".join(options)


def render_unset_binding_lines() -> list[str]:
    return [
        "unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE",
        "unset SLURM_MEM_BIND SLURM_MEM_BIND_LIST SLURM_MEM_BIND_TYPE SLURM_MEM_BIND_VERBOSE",
    ]


def should_force_torchrun(args: argparse.Namespace, dynamic_allocation: bool = False) -> bool:
    if args.launch_method == "force-torchrun":
        return True
    if args.launch_method == "plain":
        return False
    if dynamic_allocation:
        return True
    world_size = args.nodes * args.nproc_per_node
    return world_size > 1


def render_launch_exports(
    args: argparse.Namespace,
    paths: dict[str, Path],
    nnodes_value: str,
    nproc_per_node_value: str,
    master_addr_value: str,
    master_port_value: str,
    force_torchrun: bool,
) -> list[str]:
    env_exports = parse_env_pairs(args.export_env)
    launch_exports = [
        f"export LOG_DIR={shell_quote(paths['log_dir'])}",
        f"export WORKDIR={shell_quote(paths['workdir'])}",
        f"export CONFIG_FILE={shell_quote(paths['runtime_config'])}",
        "export PYTHONUNBUFFERED=1",
        "export TOKENIZERS_PARALLELISM=false",
        f"export NNODES={nnodes_value}",
        f"export NPROC_PER_NODE={nproc_per_node_value}",
        f"export MASTER_ADDR={master_addr_value}",
        f"export MASTER_PORT={master_port_value}",
    ]
    if force_torchrun:
        launch_exports.append("export FORCE_TORCHRUN=1")
    else:
        launch_exports.append("unset FORCE_TORCHRUN")

    launch_exports.extend(
        [
            f"export NCCL_IB_DISABLE={args.nccl_ib_disable}",
            f"export NCCL_SOCKET_IFNAME={shell_quote(args.nccl_socket_ifname)}",
            f"export NCCL_DEBUG={args.nccl_debug}",
            f"export TORCH_DISTRIBUTED_DEBUG={args.torch_distributed_debug}",
        ]
    )
    launch_exports.extend(render_export_lines(env_exports))
    return launch_exports


def render_training_inner_script(args: argparse.Namespace, paths: dict[str, Path]) -> str:
    train_command = build_train_command(args.train_command, paths["runtime_config"], paths["workdir"])
    pre_commands = render_pre_commands(args.pre_command)
    module_lines = [f"module load {module}" for module in args.module]

    inner_lines = [
        "set -euo pipefail",
        "cd \"${WORKDIR}\"",
        "NODE_LOG=\"${LOG_DIR}/node-${SLURM_NODEID:-unknown}.log\"",
        "exec > >(tee -a \"${NODE_LOG}\") 2>&1",
        *module_lines,
        *pre_commands,
        f"source {shell_quote(paths['env_dir'] / 'bin' / 'activate')}",
        "export NODE_RANK=${SLURM_NODEID}",
        "",
        "echo '=============================='",
        "echo \"Job id: ${SLURM_JOB_ID}\"",
        "echo \"Running on node: $(hostname)\"",
        "echo \"SLURM_NODEID: ${SLURM_NODEID}\"",
        "echo \"NODE_RANK: ${NODE_RANK}\"",
        "echo \"NNODES: ${NNODES}\"",
        "echo \"NPROC_PER_NODE: ${NPROC_PER_NODE}\"",
        "echo \"MASTER_ADDR: ${MASTER_ADDR}\"",
        "echo \"MASTER_PORT: ${MASTER_PORT}\"",
        "echo \"CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}\"",
        "echo \"Python: $(which python)\"",
        "python --version",
        "echo '=============================='",
        "",
        "python - <<'PY'",
        "import torch",
        "print('torch:', torch.__version__)",
        "print('torch compiled cuda:', torch.version.cuda)",
        "print('cuda available:', torch.cuda.is_available())",
        "if torch.cuda.is_available():",
        "    print('gpu count:', torch.cuda.device_count())",
        "    for i in range(torch.cuda.device_count()):",
        "        print(i, torch.cuda.get_device_name(i))",
        "    print('bf16 supported:', torch.cuda.is_bf16_supported())",
        "PY",
        "",
        "echo 'Training command:'",
        f"echo {shell_quote(train_command)}",
        train_command,
    ]
    return "\n".join(inner_lines)


def render_sbatch_script(args: argparse.Namespace, paths: dict[str, Path]) -> str:
    use_force_torchrun = should_force_torchrun(args)

    extra_sbatch = normalize_sbatch_extra(args.sbatch_extra)

    cpus_per_task = args.cpus_per_task or args.cpus_per_gpu * args.gpus_per_node
    sbatch_lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={args.job_name}",
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --nodes={args.nodes}",
        "#SBATCH --ntasks-per-node=1",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --output={paths['log_dir']}/slurm-%j.out",
        f"#SBATCH --error={paths['log_dir']}/slurm-%j.err",
    ]
    gres_value = args.gres if args.gres is not None else f"gpu:{args.gpus_per_node}"
    if gres_value.lower() != "none":
        sbatch_lines.append(f"#SBATCH --gres={gres_value}")
    if args.mem:
        sbatch_lines.append(f"#SBATCH --mem={args.mem}")
    if args.account:
        sbatch_lines.append(f"#SBATCH --account={args.account}")
    if args.qos:
        sbatch_lines.append(f"#SBATCH --qos={args.qos}")
    if args.reservation:
        sbatch_lines.append(f"#SBATCH --reservation={args.reservation}")
    if args.constraint:
        sbatch_lines.append(f"#SBATCH --constraint={args.constraint}")
    if args.nodelist:
        sbatch_lines.append(f"#SBATCH --nodelist={args.nodelist}")
    if args.exclude:
        sbatch_lines.append(f"#SBATCH --exclude={args.exclude}")
    sbatch_lines.extend(extra_sbatch)

    launch_exports = render_launch_exports(
        args,
        paths,
        f"${{SLURM_JOB_NUM_NODES:-{args.nodes}}}",
        str(args.nproc_per_node),
        "$(scontrol show hostnames \"${SLURM_JOB_NODELIST}\" | head -n 1)",
        "$((10000 + SLURM_JOB_ID % 50000))",
        use_force_torchrun,
    )
    inner_script = render_training_inner_script(args, paths)
    srun_options = render_srun_common_options(args)

    srun_command = (
        "srun --label "
        + (srun_options + " " if srun_options else "")
        + f"--nodes={args.nodes} "
        + f"--ntasks={args.nodes} "
        + "--ntasks-per-node=1 "
        + "bash -lc "
        + shell_quote(inner_script)
    )

    body = [
        "",
        "set -euo pipefail",
        "",
        *render_unset_binding_lines(),
        "",
        *launch_exports,
        "",
        "mkdir -p \"${LOG_DIR}\"",
        "exec > >(tee -a \"${LOG_DIR}/job-${SLURM_JOB_ID}.log\") 2>&1",
        "",
        "echo \"Log dir: ${LOG_DIR}\"",
        "echo \"Workdir: ${WORKDIR}\"",
        "echo \"Training config: ${CONFIG_FILE}\"",
        "echo \"Nodes: ${NNODES}\"",
        "echo \"GPUs per node: " + str(args.gpus_per_node) + "\"",
        "echo \"Processes per node: ${NPROC_PER_NODE}\"",
        "echo \"Launch method: " + args.launch_method + "\"",
        "echo \"ZeRO stage: " + str(args.zero_stage) + "\"",
        "echo \"Allocated hosts:\"",
        "scontrol show hostnames \"${SLURM_JOB_NODELIST}\"",
        "",
        srun_command,
    ]

    return "\n".join(sbatch_lines + body).rstrip() + "\n"


def render_existing_allocation_script(args: argparse.Namespace, paths: dict[str, Path]) -> str:
    use_force_torchrun = should_force_torchrun(args, dynamic_allocation=True)
    fallback_nnodes = args.nodes or 1
    fallback_nproc = args.nproc_per_node or args.gpus_per_node or 1

    infer_nproc_lines = [
        "ALLOC_NPROC_PER_NODE=$(printf '%s\\n' \"${JOB_INFO}\" | sed -nE 's/.*TresPerNode=[^ ]*gres\\/gpu(:[^:=]+)?:([0-9]+).*/\\2/p' | head -n 1)",
        "if [ -z \"${ALLOC_NPROC_PER_NODE}\" ]; then",
        "  ALLOC_GPU_TOTAL=$(printf '%s\\n' \"${JOB_INFO}\" | sed -nE 's/.*TRES=[^ ]*gres\\/gpu=([0-9]+).*/\\1/p' | head -n 1)",
        "  if [ -n \"${ALLOC_GPU_TOTAL}\" ] && [ -n \"${ALLOC_NNODES}\" ] && [ \"${ALLOC_NNODES}\" -gt 0 ]; then",
        "    ALLOC_NPROC_PER_NODE=$(( (ALLOC_GPU_TOTAL + ALLOC_NNODES - 1) / ALLOC_NNODES ))",
        "  fi",
        "fi",
        "if [ -z \"${ALLOC_NPROC_PER_NODE}\" ]; then",
        "  ALLOC_NPROC_PER_NODE=$(printf '%s\\n' \"${JOB_INFO}\" | sed -nE 's/.*Gres=[^ ]*gpu(:[^: ]+)?:([0-9]+).*/\\2/p' | head -n 1)",
        "fi",
        f"if [ -z \"${{ALLOC_NPROC_PER_NODE}}\" ]; then ALLOC_NPROC_PER_NODE={fallback_nproc}; fi",
        "export ALLOC_NPROC_PER_NODE",
    ]
    if args.explicit_nproc_per_node:
        infer_nproc_lines = [f"export ALLOC_NPROC_PER_NODE={args.nproc_per_node}"]
    elif args.explicit_gpus_per_node:
        infer_nproc_lines = [f"export ALLOC_NPROC_PER_NODE={args.gpus_per_node}"]

    launch_exports = render_launch_exports(
        args,
        paths,
        "${ALLOC_NNODES}",
        "${ALLOC_NPROC_PER_NODE}",
        "$(scontrol show hostnames \"${ALLOC_NODELIST}\" | head -n 1)",
        "$((10000 + ALLOCATION_JOB_ID % 50000))",
        use_force_torchrun,
    )
    inner_script = render_training_inner_script(args, paths)
    srun_options = render_srun_common_options(args)
    srun_command = (
        "srun --jobid=\"${ALLOCATION_JOB_ID}\" --label "
        + (srun_options + " " if srun_options else "")
        + "--nodes=\"${NNODES}\" "
        + "--ntasks=\"${NNODES}\" "
        + "--ntasks-per-node=1 "
        + "bash -lc "
        + shell_quote(inner_script)
    )

    body = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        *render_unset_binding_lines(),
        "",
        f"export ALLOCATION_JOB_ID={shell_quote(args.jobid)}",
        "JOB_INFO=$(scontrol show job \"${ALLOCATION_JOB_ID}\")",
        "ALLOC_NODELIST=$(printf '%s\\n' \"${JOB_INFO}\" | sed -nE 's/.*NodeList=([^ ]+).*/\\1/p' | head -n 1)",
        "if [ -z \"${ALLOC_NODELIST}\" ] || [ \"${ALLOC_NODELIST}\" = \"(null)\" ]; then",
        "  echo \"Could not resolve NodeList for Slurm allocation ${ALLOCATION_JOB_ID}\" >&2",
        "  echo \"${JOB_INFO}\" >&2",
        "  exit 2",
        "fi",
        "export ALLOC_NODELIST",
        "ALLOC_NNODES=$(printf '%s\\n' \"${JOB_INFO}\" | sed -nE 's/.*NumNodes=([0-9]+).*/\\1/p' | head -n 1)",
        f"if [ -z \"${{ALLOC_NNODES}}\" ]; then ALLOC_NNODES={fallback_nnodes}; fi",
        "export ALLOC_NNODES",
        *infer_nproc_lines,
        "",
        *launch_exports,
        "",
        "mkdir -p \"${LOG_DIR}\"",
        "exec > >(tee -a \"${LOG_DIR}/existing-allocation-${ALLOCATION_JOB_ID}.log\") 2>&1",
        "",
        "echo \"Using existing Slurm allocation: ${ALLOCATION_JOB_ID}\"",
        "echo \"Log dir: ${LOG_DIR}\"",
        "echo \"Workdir: ${WORKDIR}\"",
        "echo \"Training config: ${CONFIG_FILE}\"",
        "echo \"Nodes: ${NNODES}\"",
        "echo \"Processes per node: ${NPROC_PER_NODE}\"",
        "echo \"Launch method: " + args.launch_method + "\"",
        "echo \"ZeRO stage: " + str(args.zero_stage) + "\"",
        "echo \"Allocated hosts:\"",
        "scontrol show hostnames \"${ALLOC_NODELIST}\"",
        "",
        srun_command,
    ]
    return "\n".join(body).rstrip() + "\n"


def create_run_files(args: argparse.Namespace) -> dict[str, Path]:
    config = resolve_existing_path(args.config, "training config")
    env_dir = resolve_existing_path(args.env_dir, "environment directory")
    activate = env_dir / "bin" / "activate"
    if not activate.exists():
        raise SystemExit(f"environment activate script does not exist: {activate}")

    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else config.parent
    if not workdir.exists():
        raise SystemExit(f"workdir does not exist: {workdir}")

    runs_dir = Path(args.runs_dir).expanduser().resolve()
    run_name = f"{sanitize_name(args.job_name)}_{timestamp()}"
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    if args.log_dir:
        log_root = Path(args.log_dir).expanduser().resolve()
        log_root.mkdir(parents=True, exist_ok=True)
        log_dir = log_root / run_name
        log_dir.mkdir(parents=True, exist_ok=False)
    else:
        log_dir = run_dir

    deepspeed_config: Path | None = None
    if args.zero_stage in (2, 3):
        if args.deepspeed_dir:
            if args.deepspeed_template:
                raise SystemExit("Use either --deepspeed-dir or --deepspeed-template, not both")
            deepspeed_config = resolve_deepspeed_config_from_dir(
                args.deepspeed_dir,
                args.zero_stage,
                args.deepspeed_config_name,
            )
        elif args.deepspeed_template:
            template = resolve_existing_path(args.deepspeed_template, "DeepSpeed template")
            deepspeed_config = run_dir / template.name
            shutil.copy2(template, deepspeed_config)
        else:
            deepspeed_config = run_dir / f"deepspeed_zero{args.zero_stage}.json"
            deepspeed_config.write_text(
                json.dumps(make_deepspeed_config(args.zero_stage), indent=2) + "\n",
                encoding="utf-8",
            )

    runtime_config = run_dir / config.name
    write_runtime_config(config, runtime_config, args.zero_stage, deepspeed_config)

    paths = {
        "config": config,
        "runtime_config": runtime_config,
        "workdir": workdir,
        "env_dir": env_dir,
        "run_dir": run_dir,
        "log_dir": log_dir,
    }
    if deepspeed_config:
        paths["deepspeed_config"] = deepspeed_config

    if args.jobid:
        allocation_script = run_dir / "run_existing_allocation.sh"
        allocation_script.write_text(render_existing_allocation_script(args, paths), encoding="utf-8")
        allocation_script.chmod(0o755)
        paths["allocation_script"] = allocation_script
    else:
        sbatch_script = run_dir / "submit.sbatch"
        sbatch_script.write_text(render_sbatch_script(args, paths), encoding="utf-8")
        paths["sbatch_script"] = sbatch_script
    return paths


def write_launcher_log(paths: dict[str, Path], args: argparse.Namespace) -> Path:
    launcher_log = paths["log_dir"] / "launcher.log"
    lines = [
        f"created_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"artifact_dir: {paths['run_dir']}",
        f"log_dir: {paths['log_dir']}",
        f"runtime_config: {paths['runtime_config']}",
        f"jobid: {args.jobid or ''}",
        f"wait: {args.wait}",
        f"dry_run: {args.dry_run}",
        f"nodes: {args.nodes}",
        f"gpus_per_node: {args.gpus_per_node}",
        f"nproc_per_node: {args.nproc_per_node}",
        f"zero_stage: {args.zero_stage}",
    ]
    if "sbatch_script" in paths:
        lines.append(f"sbatch_script: {paths['sbatch_script']}")
    if "allocation_script" in paths:
        lines.append(f"allocation_script: {paths['allocation_script']}")
    if "deepspeed_config" in paths:
        lines.append(f"deepspeed_config: {paths['deepspeed_config']}")
    launcher_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return launcher_log


def submit_job(sbatch_script: Path, submit_log: Path, wait: bool) -> int:
    command = ["sbatch"]
    if wait:
        command.append("--wait")
    command.append(str(sbatch_script))

    submit_log.parent.mkdir(parents=True, exist_ok=True)
    command_text = " ".join(shell_quote(part) for part in command)
    with submit_log.open("w", encoding="utf-8") as log_file:
        def emit(message: str) -> None:
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        emit("Submitting job:")
        emit(command_text)
        if wait:
            emit("Waiting for Slurm job to finish...")

        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if process.stdout is not None:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()
        returncode = process.wait()
        if wait:
            emit(f"Slurm job finished with sbatch exit code {returncode}.")
        return returncode


def run_existing_allocation(allocation_script: Path, run_log: Path) -> int:
    command = ["bash", str(allocation_script)]
    run_log.parent.mkdir(parents=True, exist_ok=True)
    command_text = " ".join(shell_quote(part) for part in command)
    with run_log.open("w", encoding="utf-8") as log_file:
        def emit(message: str) -> None:
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        emit("Running inside existing Slurm allocation:")
        emit(command_text)

        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if process.stdout is not None:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()
        returncode = process.wait()
        emit(f"Existing-allocation run finished with exit code {returncode}.")
        return returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and submit Slurm sbatch jobs for LLaMA-Factory training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to the LLaMA-Factory YAML config.")
    parser.add_argument("--env-dir", required=True, help="Python virtualenv/conda env directory containing bin/activate.")
    parser.add_argument("--workdir", help="Directory to run training from. Defaults to the config directory.")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR), help="Directory for generated runtime files and sbatch scripts.")
    parser.add_argument(
        "--log-dir",
        help="Directory for logs. A per-run subdirectory is created here. Defaults to the generated run directory.",
    )

    parser.add_argument("--job-name", default="llamafactory_train", help="Slurm job name.")
    parser.add_argument(
        "--jobid",
        nargs="?",
        const="",
        help="Existing Slurm allocation/job id from salloc. If used without a value, SLURM_JOB_ID is used.",
    )
    parser.add_argument("--partition", help="Slurm partition. Required unless --jobid is used.")
    parser.add_argument("--nodes", type=positive_int, help="Number of nodes. Defaults to 1 in sbatch mode; inferred in --jobid mode.")
    parser.add_argument(
        "--gpus-per-node",
        type=positive_int,
        help="GPUs allocated on each node. Defaults to 1 in sbatch mode; inferred in --jobid mode when possible.",
    )
    parser.add_argument(
        "--gres",
        help="Slurm GRES value. Defaults to gpu:<gpus-per-node>. Use 'none' and --sbatch-extra for clusters that prefer another GPU flag.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=positive_int,
        help="Trainer processes per node. Defaults to --gpus-per-node.",
    )
    parser.add_argument("--cpus-per-gpu", type=positive_int, default=8, help="CPUs per GPU if --cpus-per-task is not set.")
    parser.add_argument("--cpus-per-task", type=positive_int, help="CPUs for each Slurm launcher task.")
    parser.add_argument("--mem", default="120G", help="Memory per node, for Slurm --mem.")
    parser.add_argument("--time", default="12:00:00", help="Slurm time limit.")
    parser.add_argument("--account", help="Slurm account.")
    parser.add_argument("--qos", help="Slurm QOS.")
    parser.add_argument("--reservation", help="Slurm reservation.")
    parser.add_argument("--constraint", help="Slurm node constraint.")
    parser.add_argument("--nodelist", help="Slurm nodelist.")
    parser.add_argument("--exclude", help="Slurm exclude list.")
    parser.add_argument(
        "--sbatch-extra",
        action="append",
        default=[],
        help="Extra sbatch option. Repeatable. Example: --sbatch-extra account=myacct",
    )

    parser.add_argument(
        "--zero-stage",
        type=int,
        choices=(0, 2, 3),
        default=0,
        help="DeepSpeed ZeRO stage. 0 removes the top-level deepspeed key in the runtime YAML.",
    )
    parser.add_argument(
        "--deepspeed-template",
        help="Existing DeepSpeed JSON to copy for ZeRO-2/ZeRO-3.",
    )
    parser.add_argument(
        "--deepspeed-dir",
        help="Directory containing an existing DeepSpeed JSON to use for ZeRO-2/ZeRO-3.",
    )
    parser.add_argument(
        "--deepspeed-config-name",
        help="DeepSpeed JSON filename inside --deepspeed-dir. Supports {stage}.",
    )
    parser.add_argument(
        "--launch-method",
        choices=("auto", "force-torchrun", "plain"),
        default="auto",
        help="auto uses FORCE_TORCHRUN when world size is greater than 1.",
    )
    parser.add_argument(
        "--train-command",
        help="Override training command. Supports {config} and {workdir}. Default: llamafactory-cli train {config}",
    )
    parser.add_argument(
        "--module",
        action="append",
        default=[],
        help="Module to load before training. Repeatable.",
    )
    parser.add_argument(
        "--pre-command",
        action="append",
        default=[],
        help="Command to run before environment activation/training. Repeatable.",
    )
    parser.add_argument(
        "--export-env",
        action="append",
        default=[],
        help="Environment override KEY=VALUE exported in the job. Repeatable.",
    )
    parser.add_argument(
        "--srun-cpu-bind",
        default="none",
        help="CPU binding passed to generated srun steps. Use 'default' to omit --cpu-bind.",
    )

    parser.add_argument("--nccl-ib-disable", default="1", help="Value for NCCL_IB_DISABLE.")
    parser.add_argument(
        "--nccl-socket-ifname",
        default="^lo,docker0,virbr0",
        help="Value for NCCL_SOCKET_IFNAME.",
    )
    parser.add_argument("--nccl-debug", default="INFO", help="Value for NCCL_DEBUG.")
    parser.add_argument(
        "--torch-distributed-debug",
        default="DETAIL",
        help="Value for TORCH_DISTRIBUTED_DEBUG.",
    )

    wait_group = parser.add_mutually_exclusive_group()
    wait_group.add_argument(
        "--wait",
        dest="wait",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Submit with sbatch --wait so this launcher exits after the training job finishes.",
    )
    wait_group.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Submit with plain sbatch and return immediately after the job is accepted.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only write files; do not call sbatch.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.explicit_gpus_per_node = args.gpus_per_node is not None
    args.explicit_nproc_per_node = args.nproc_per_node is not None
    if not hasattr(args, "wait"):
        args.wait = True
    if args.jobid == "":
        args.jobid = os.environ.get("SLURM_JOB_ID")
        if not args.jobid:
            raise SystemExit("--jobid needs a value unless SLURM_JOB_ID is set")
    if not args.jobid and not args.partition:
        raise SystemExit("--partition is required unless --jobid is used")
    if args.jobid and not args.wait:
        raise SystemExit("--no-wait only applies to sbatch mode; --jobid mode runs the srun step in the foreground")
    if args.gpus_per_node is None:
        args.gpus_per_node = 1
    if args.nodes is None:
        args.nodes = 1
    if args.nproc_per_node is None:
        args.nproc_per_node = args.gpus_per_node
    if args.nproc_per_node > args.gpus_per_node and (not args.jobid or args.explicit_gpus_per_node):
        raise SystemExit("--nproc-per-node cannot be larger than --gpus-per-node")
    if args.zero_stage == 0 and args.deepspeed_template:
        raise SystemExit("--deepspeed-template only applies when --zero-stage is 2 or 3")
    if args.zero_stage == 0 and args.deepspeed_dir:
        raise SystemExit("--deepspeed-dir only applies when --zero-stage is 2 or 3")
    if args.zero_stage == 0 and args.deepspeed_config_name:
        raise SystemExit("--deepspeed-config-name only applies when --zero-stage is 2 or 3")

    paths = create_run_files(args)
    launcher_log = write_launcher_log(paths, args)
    print(f"Run dir: {paths['run_dir']}")
    print(f"Log dir: {paths['log_dir']}")
    print(f"Launcher log: {launcher_log}")
    print(f"Runtime config: {paths['runtime_config']}")
    if "deepspeed_config" in paths:
        print(f"DeepSpeed config: {paths['deepspeed_config']}")
    if "sbatch_script" in paths:
        print(f"sbatch script: {paths['sbatch_script']}")
    if "allocation_script" in paths:
        print(f"allocation script: {paths['allocation_script']}")

    if args.dry_run:
        print("Dry run: not submitting job.")
        return 0

    if args.jobid:
        rc = run_existing_allocation(paths["allocation_script"], paths["log_dir"] / "existing-allocation-submit.log")
        if rc != 0:
            print(f"existing allocation run failed; see {paths['log_dir'] / 'existing-allocation-submit.log'}", file=sys.stderr)
    else:
        rc = submit_job(paths["sbatch_script"], paths["log_dir"] / "sbatch-submit.log", args.wait)
        if rc != 0:
            print(f"sbatch failed; see {paths['log_dir'] / 'sbatch-submit.log'}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
