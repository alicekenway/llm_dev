#!/usr/bin/env python3
"""Export a LLaMA-Factory LoRA adapter as a merged Hugging Face model."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any


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


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_existing_path(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"{label} does not exist: {path}")
    return path


def ensure_output_target(path: Path, allow_existing: bool) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        return
    if not path.is_dir():
        raise SystemExit(f"output path exists and is not a directory: {path}")
    if any(path.iterdir()) and not allow_existing:
        raise SystemExit(
            f"output directory is not empty: {path}\n"
            "Use --allow-existing-output if you intentionally want LLaMA-Factory to write there."
        )


def json_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_set_item(item: str) -> tuple[list[str], Any]:
    if "=" not in item:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {item!r}")
    raw_key, raw_value = item.split("=", 1)
    keys = [part for part in raw_key.split(".") if part]
    if not keys:
        raise argparse.ArgumentTypeError(f"expected non-empty KEY in {item!r}")
    return keys, json_scalar(raw_value)


def set_deep(mapping: dict[str, Any], keys: list[str], value: Any) -> None:
    target = mapping
    for key in keys[:-1]:
        child = target.setdefault(key, {})
        if not isinstance(child, dict):
            raise SystemExit(f"cannot set {'.'.join(keys)} because {key!r} is not a mapping")
        target = child
    target[keys[-1]] = value


def first_simple_yaml_value(path: Path, key: str) -> str | None:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.*?)\s*(?:#.*)?$")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return None
    for line in lines:
        match = pattern.match(line)
        if not match:
            continue
        value = match.group(1).strip()
        if not value:
            return None
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        return value
    return None


def find_training_config(adapter: Path) -> Path | None:
    names = ["sft.yaml", "sft.yml", "train.yaml", "train.yml", "training.yaml", "training.yml"]
    search_parents = [adapter, *list(adapter.parents)[:8]]
    for parent in search_parents:
        matches = [parent / name for name in names if (parent / name).exists()]
        if matches:
            return matches[0]

    for parent in search_parents:
        if parent == parent.parent:
            continue
        candidates = sorted(parent.glob("*.yaml")) + sorted(parent.glob("*.yml"))
        for candidate in candidates:
            if first_simple_yaml_value(candidate, "template"):
                return candidate
    return None


def resolve_template(args: argparse.Namespace, adapter: Path) -> str | None:
    if args.template:
        return args.template
    if args.no_template_auto_detect:
        return None

    training_config = Path(args.training_config).expanduser().resolve() if args.training_config else find_training_config(adapter)
    if training_config is None:
        return None
    if not training_config.exists():
        raise SystemExit(f"training config does not exist: {training_config}")

    template = first_simple_yaml_value(training_config, "template")
    if template:
        print(f"Detected template from {training_config}: {template}")
    return template


def build_export_config(args: argparse.Namespace, base_model: Path, adapter: Path, output_dir: Path) -> dict[str, Any]:
    template = resolve_template(args, adapter)
    if args.config_format == "v1":
        config: dict[str, Any] = {
            "model": str(base_model),
            "peft_config": {
                "name": "lora",
                "adapter_name_or_path": str(adapter),
                "export_dir": str(output_dir),
                "export_size": args.export_size,
                "infer_dtype": args.infer_dtype,
            },
        }
        if template:
            config["template"] = template
        if args.trust_remote_code:
            config["trust_remote_code"] = True
    else:
        config = {
            "model_name_or_path": str(base_model),
            "adapter_name_or_path": str(adapter),
            "finetuning_type": "lora",
            "export_dir": str(output_dir),
            "export_size": args.export_size,
            "export_device": args.export_device,
            "infer_dtype": args.infer_dtype,
            "trust_remote_code": args.trust_remote_code,
        }
        if template:
            config["template"] = template

    for keys, value in args.set:
        set_deep(config, keys, value)
    return config


def resolve_cli_command(args: argparse.Namespace, *, require: bool) -> list[str]:
    if args.llamafactory_cli:
        cli = Path(args.llamafactory_cli).expanduser()
        return [str(cli)]

    if args.env_dir:
        env_dir = resolve_existing_path(args.env_dir, "environment directory")
        cli = env_dir / "bin" / "llamafactory-cli"
        if cli.exists():
            return [str(cli)]
        python = env_dir / "bin" / "python"
        if python.exists():
            return [str(python), "-m", "llamafactory.cli"]
        raise SystemExit(f"could not find llamafactory-cli or python under: {env_dir / 'bin'}")

    cli = shutil.which("llamafactory-cli")
    if not cli:
        if not require:
            return ["llamafactory-cli"]
        raise SystemExit("could not find llamafactory-cli. Pass --env-dir or --llamafactory-cli.")
    return [cli]


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_export_output(output_dir: Path) -> None:
    required = ["config.json", "tokenizer_config.json"]
    missing = [name for name in required if not (output_dir / name).exists()]
    weight_files = list(output_dir.glob("*.safetensors")) + list(output_dir.glob("*.bin"))
    if missing:
        raise SystemExit(f"export finished, but required file(s) are missing in {output_dir}: {', '.join(missing)}")
    if not weight_files:
        raise SystemExit(f"export finished, but no model weight files were found in {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a LLaMA-Factory LoRA adapter as a merged model.")
    parser.add_argument("--base-model", required=True, help="Base model path used for training.")
    parser.add_argument("--adapter", required=True, help="LLaMA-Factory LoRA adapter directory.")
    parser.add_argument("--output-dir", required=True, help="Directory where the merged model will be written.")
    parser.add_argument("--env-dir", help="Python environment containing bin/llamafactory-cli.")
    parser.add_argument("--llamafactory-cli", help="Explicit llamafactory-cli executable path.")
    parser.add_argument(
        "--config-format",
        choices=["v1", "classic"],
        default="classic",
        help="Generated LLaMA-Factory export config format. Default: classic.",
    )
    parser.add_argument("--template", help="Optional LLaMA-Factory template name.")
    parser.add_argument("--training-config", help="Training YAML to read template from when --template is omitted.")
    parser.add_argument(
        "--no-template-auto-detect",
        action="store_true",
        help="Do not search adapter parent directories for a training YAML with template.",
    )
    parser.add_argument(
        "--infer-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Dtype used by LLaMA-Factory during export.",
    )
    parser.add_argument(
        "--export-device",
        choices=["auto", "cpu"],
        default="auto",
        help="Classic config export_device value. v1 export ignores this.",
    )
    parser.add_argument("--export-size", type=positive_int, default=5, help="Shard size in GB. Default: 5.")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set trust_remote_code in the generated config. Default: true.",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow LLaMA-Factory to write into a non-empty output directory.",
    )
    parser.add_argument(
        "--run-dir",
        help=f"Directory for generated config and command metadata. Default: {DEFAULT_RUNS_DIR}/export_<timestamp>.",
    )
    parser.add_argument(
        "--set",
        action="append",
        type=parse_set_item,
        default=[],
        metavar="KEY=VALUE",
        help="Override generated config values. Supports dotted keys for nested config values.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write config and print command without running export.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_model = resolve_existing_path(args.base_model, "base model")
    adapter = resolve_existing_path(args.adapter, "adapter")
    if not adapter.is_dir():
        raise SystemExit(f"adapter must be a directory: {adapter}")
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"adapter_config.json was not found in adapter directory: {adapter}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_target(output_dir, args.allow_existing_output)

    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else DEFAULT_RUNS_DIR / f"export_{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    config = build_export_config(args, base_model, adapter, output_dir)
    config_path = run_dir / "export_config.yaml"
    write_json(config_path, config)

    command = [*resolve_cli_command(args, require=not args.dry_run), "export", str(config_path)]
    write_json(
        run_dir / "export_command.json",
        {
            "command": command,
            "command_text": command_text(command),
            "base_model": str(base_model),
            "adapter": str(adapter),
            "output_dir": str(output_dir),
            "config_format": args.config_format,
        },
    )

    print(f"Export config: {config_path}")
    print(f"Command: {command_text(command)}")
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    result = subprocess.run(command, cwd=str(run_dir), env=env)
    if result.returncode != 0:
        return result.returncode

    validate_export_output(output_dir)
    print(f"Merged model: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
