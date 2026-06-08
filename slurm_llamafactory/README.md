# LLaMA-Factory Slurm Launcher

This directory contains `llamafactory_slurm_launcher.py`, a small Python wrapper that generates an `sbatch` script for LLaMA-Factory training, creates a per-run log directory, optionally patches the runtime YAML with a DeepSpeed ZeRO config, and submits the job.

The original training YAML is never modified. Generated runtime files go under `runs/` by default. Logs can be sent elsewhere with `--log-dir`.

## What It Handles

- Single GPU: `--nodes 1 --gpus-per-node 1`
- Single-node multi-GPU: `--nodes 1 --gpus-per-node N`
- Multi-node multi-GPU: `--nodes N --gpus-per-node G`
- Existing `salloc` allocation: `--jobid <allocation_id>`
- ZeRO-0, ZeRO-2, ZeRO-3: `--zero-stage 0|2|3`
- Slurm options: partition, account, QOS, reservation, constraints, memory, time, nodelist, exclude list, and arbitrary extra `#SBATCH` lines
- GPU allocation through Slurm `--gres` by default, with override support for clusters that use different GPU flags
- Log collection under the per-run log directory:
  - `launcher.log`
  - `slurm-%j.out`
  - `slurm-%j.err`
  - `job-%j.log`
  - `node-<SLURM_NODEID>.log`
  - `sbatch-submit.log`
  - `existing-allocation-submit.log` when `--jobid` is used
- Blocking submission by default with `sbatch --wait`
- Generated `srun` steps default to `--cpu-bind=none` to avoid inherited CPU binding masks

## Basic Usage

```bash
python3 llamafactory_slurm_launcher.py \
  --config /path/to/sft.yaml \
  --env-dir /mnt/users/jinyang_wang/LLM/llamafactory_env \
  --partition h200.141gb \
  --log-dir /mnt/users/jinyang_wang/LLM/training_logs \
  --nodes 1 \
  --gpus-per-node 1 \
  --zero-stage 0
```

By default the launcher submits the generated script with `sbatch --wait`, so the Python process waits until the Slurm training job finishes. Use `--no-wait` to submit and return immediately. Use `--dry-run` to only generate files.

If you already created an allocation with `salloc`, use `--jobid <allocation_id>` instead of `--partition`. In that mode the launcher does not call `sbatch`; it runs `srun --jobid=<allocation_id>` in the foreground.

Do not prefix normal `sbatch` mode with `srun`. Run the launcher directly:

```bash
python3 llamafactory_slurm_launcher.py ...
```

not:

```bash
srun python3 llamafactory_slurm_launcher.py ...
```

If you already have an allocation, use `--jobid` instead of wrapping the launcher with `srun`.

When `--log-dir /path/to/logs` is provided, the launcher creates a per-run log directory below that path:

```text
/path/to/logs/<job_name>_<timestamp>/
```

```bash
python3 llamafactory_slurm_launcher.py \
  --config ./sft.yaml \
  --env-dir /mnt/users/jinyang_wang/LLM/llamafactory_env \
  --partition h200.141gb \
  --nodes 1 \
  --gpus-per-node 2 \
  --zero-stage 2 \
  --dry-run
```

## Common Examples

### 1 GPU, No DeepSpeed

```bash
python3 llamafactory_slurm_launcher.py \
  --config ./sft.yaml \
  --env-dir /mnt/users/jinyang_wang/LLM/llamafactory_env \
  --partition h200.141gb \
  --job-name qwen3_lora_1gpu \
  --log-dir /mnt/users/jinyang_wang/LLM/training_logs \
  --nodes 1 \
  --gpus-per-node 1 \
  --zero-stage 0
```

### 1 Node, 2 GPUs, ZeRO-2

```bash
python3 llamafactory_slurm_launcher.py \
  --config ./sft.yaml \
  --env-dir /mnt/users/jinyang_wang/LLM/llamafactory_env \
  --partition h200.141gb \
  --job-name qwen3_lora_1node_2gpu_z2 \
  --log-dir /mnt/users/jinyang_wang/LLM/training_logs \
  --nodes 1 \
  --gpus-per-node 2 \
  --zero-stage 2
```

The launcher allocates 2 GPUs on one node, starts one Slurm task on that node, and lets LLaMA-Factory launch 2 local torch processes with:

```bash
FORCE_TORCHRUN=1
NNODES=1
NPROC_PER_NODE=2
```

## Launch Structure

Normal `sbatch` mode has one `srun` inside the generated `sbatch` script:

```text
python launcher -> sbatch --wait submit.sbatch -> srun one task per node -> llamafactory-cli train
```

That is not nested `srun` in the normal `sbatch` case. Slurm runs the batch script on the batch host, then the script uses `srun` to create one distributed job step across the allocated nodes. LLaMA-Factory then uses `FORCE_TORCHRUN`, `NNODES`, `NODE_RANK`, `NPROC_PER_NODE`, `MASTER_ADDR`, and `MASTER_PORT` to start the actual torch workers.

If you run the launcher itself from inside another `srun`, it will still submit a separate Slurm job with `sbatch`. In that case the outer `srun` is only wrapping the submission command, not the training job.

Avoid that pattern when possible. An outer `srun` can export CPU-binding variables like `SLURM_CPU_BIND_LIST`; those masks may not be valid for the generated training job step. The launcher now unsets inherited CPU/memory binding variables and uses `--srun-cpu-bind none` by default, but the cleaner workflow is direct `python ...` for `sbatch` mode or `python ... --jobid` for an existing allocation.

To submit without blocking:

```bash
--no-wait
```

Existing-allocation mode uses a preallocated Slurm job:

```text
salloc -> python launcher --jobid <allocation_id> -> srun --jobid <allocation_id> -> llamafactory-cli train
```

This mode always waits for the training `srun` step to finish.

### Existing salloc Allocation

First allocate resources:

```bash
salloc \
  -J qwen3_lora_alloc \
  --partition h200.141gb \
  --nodes 2 \
  --gres gpu:2 \
  --cpus-per-task 16 \
  --mem 240G \
  --time 12:00:00
```

After Slurm grants the allocation, note the allocation id printed by `salloc`, then run:

```bash
python3 llamafactory_slurm_launcher.py \
  --config ./sft.yaml \
  --env-dir /mnt/users/jinyang_wang/LLM/llamafactory_env \
  --jobid <allocation_id> \
  --job-name qwen3_lora_existing_alloc \
  --log-dir /mnt/users/jinyang_wang/LLM/training_logs \
  --zero-stage 2
```

The launcher queries `scontrol show job <allocation_id>` to infer `NNODES`, the allocated host list, and GPU/process count per node when possible. If the GPU count cannot be inferred on your cluster, pass it explicitly:

```bash
--jobid <allocation_id> --gpus-per-node 2
```

If you are already inside the `salloc` shell, you can omit the value and use `SLURM_JOB_ID`:

```bash
python3 llamafactory_slurm_launcher.py \
  --config ./sft.yaml \
  --env-dir /mnt/users/jinyang_wang/LLM/llamafactory_env \
  --jobid \
  --zero-stage 2
```

### 2 Nodes, 2 GPUs Per Node, ZeRO-3

```bash
python3 llamafactory_slurm_launcher.py \
  --config ./sft.yaml \
  --env-dir /mnt/users/jinyang_wang/LLM/llamafactory_env \
  --partition h200.141gb \
  --job-name qwen3_lora_2node_4gpu_z3 \
  --log-dir /mnt/users/jinyang_wang/LLM/training_logs \
  --nodes 2 \
  --gpus-per-node 2 \
  --zero-stage 3 \
  --time 24:00:00 \
  --mem 240G
```

This runs one launcher task per node. Each launcher sets:

```bash
NNODES=<number of allocated nodes>
NODE_RANK=<Slurm node id>
NPROC_PER_NODE=<GPUs per node>
MASTER_ADDR=<first allocated host>
MASTER_PORT=<stable port derived from SLURM_JOB_ID>
FORCE_TORCHRUN=1
```

## Slurm Controls

Use normal options for common Slurm fields:

```bash
--account my_account
--qos normal
--reservation my_reservation
--constraint h200
--nodelist node001,node002
--exclude node003
--cpus-per-gpu 8
--gres gpu:2
--mem 120G
--time 12:00:00
```

For uncommon Slurm flags, repeat `--sbatch-extra`:

```bash
--sbatch-extra account=my_account
--sbatch-extra mail-type=END,FAIL
--sbatch-extra mail-user=you@example.com
```

By default the launcher emits `#SBATCH --gres=gpu:<gpus-per-node>`. If your cluster uses another format, override it:

```bash
--gres gpu:h200:2
```

If your cluster does not use `--gres`, skip it and provide your own GPU allocation flag:

```bash
--gres none
--sbatch-extra gpus-per-node=2
```

## DeepSpeed Behavior

`--zero-stage 0` removes any top-level `deepspeed:` key from the runtime copy of the YAML.

`--zero-stage 2` or `--zero-stage 3` uses DeepSpeed and appends a `deepspeed:` path to the runtime YAML.

If no DeepSpeed config is provided, the launcher writes a generated JSON into the run directory:

```yaml
deepspeed: /absolute/path/to/runs/<job>/deepspeed_zero2.json
```

If you already have DeepSpeed configs in a directory, pass `--deepspeed-dir`. The launcher does not write into this directory; it loads an existing JSON from it and references that file directly.

By default it looks for one of these filenames:

```text
deepspeed_zero2.json
ds_zero2.json
zero2.json
ds_z2_config.json
z2.json
```

For ZeRO-3, replace `2` with `3`.

```bash
python3 llamafactory_slurm_launcher.py \
  --config ./sft.yaml \
  --env-dir /mnt/users/jinyang_wang/LLM/llamafactory_env \
  --partition h200.141gb \
  --nodes 1 \
  --gpus-per-node 2 \
  --zero-stage 2 \
  --deepspeed-dir /mnt/users/jinyang_wang/LLM/deepspeed_configs
```

If your filename is different:

```bash
--deepspeed-dir /mnt/users/jinyang_wang/LLM/deepspeed_configs \
--deepspeed-config-name ds_z{stage}_offload_config.json
```

To copy one explicit DeepSpeed config into the generated run directory:

```bash
python3 llamafactory_slurm_launcher.py \
  --config ./sft.yaml \
  --env-dir /mnt/users/jinyang_wang/LLM/llamafactory_env \
  --partition h200.141gb \
  --nodes 2 \
  --gpus-per-node 4 \
  --zero-stage 3 \
  --deepspeed-template ./my_ds_zero3.json
```

## Environment And Debug Options

Load modules before activation:

```bash
--module cuda/12.4
--module gcc/12
```

Run setup commands before training:

```bash
--pre-command "ulimit -n 65535"
```

Generated `srun` steps default to:

```bash
--srun-cpu-bind none
```

To let Slurm use its cluster default CPU binding:

```bash
--srun-cpu-bind default
```

Export extra environment variables:

```bash
--export-env HF_HOME=/path/to/hf_cache
--export-env WANDB_MODE=offline
```

NCCL defaults are conservative for clusters where InfiniBand is not configured correctly:

```bash
--nccl-ib-disable 1
--nccl-socket-ifname '^lo,docker0,virbr0'
--nccl-debug INFO
--torch-distributed-debug DETAIL
```

If your cluster has working IB/RDMA, try:

```bash
--nccl-ib-disable 0
--nccl-debug WARN
--torch-distributed-debug OFF
```

## Launch Method

Default behavior is `--launch-method auto`:

- world size 1: plain `llamafactory-cli train`
- world size greater than 1: `FORCE_TORCHRUN=1 llamafactory-cli train`
- `--jobid` mode: `FORCE_TORCHRUN=1` because the world size is inferred at runtime

You can force either behavior:

```bash
--launch-method force-torchrun
--launch-method plain
```

You can also replace the training command. The placeholders `{config}` and `{workdir}` are shell-quoted automatically:

```bash
--train-command "llamafactory-cli train {config}"
```

## Inspect Before Submit

Use `--dry-run`, then inspect the generated files:

```bash
python3 llamafactory_slurm_launcher.py \
  --config ./sft.yaml \
  --env-dir /mnt/users/jinyang_wang/LLM/llamafactory_env \
  --partition h200.141gb \
  --nodes 2 \
  --gpus-per-node 2 \
  --zero-stage 3 \
  --dry-run
```

The command prints the run directory and generated `submit.sbatch` path. In `--jobid` mode it prints `run_existing_allocation.sh` instead.

## No Logs Yet

The launcher creates `launcher.log`, a generated launch script, and the runtime YAML before starting Slurm work. `launcher.log` is in the per-run log directory. `submit.sbatch` or `run_existing_allocation.sh` and the runtime YAML are in the generated run directory. If those files are missing, the program failed before submission or you used a different `--runs-dir` or `--log-dir`.

`slurm-%j.out`, `slurm-%j.err`, `job-%j.log`, and `node-<rank>.log` appear only after Slurm starts the batch job. In `--jobid` mode, look for `existing-allocation-submit.log`, `existing-allocation-<jobid>.log`, and `node-<rank>.log`. If the job is still pending in the queue, training logs may not exist yet.

Make sure `--log-dir`, `--runs-dir`, `--workdir`, `--config`, and `--deepspeed-dir` point to a filesystem visible from the compute nodes. If they point to login-node local storage, the batch job may be unable to write logs.
