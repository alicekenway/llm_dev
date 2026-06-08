#!/usr/bin/env python3
"""Run model inference on LLaMA-Factory-style JSON/JSONL test data.

Input records are expected to contain fields like:
  instruction, input, output, system, history

The output preserves the input format and adds a prediction field to every
record. Statistics are intentionally handled by compute_stats.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PREDICTION_FIELD = "prediction"


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read the config file: pip install pyyaml") from exc

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"YAML config must be a mapping/object: {path}")
    return data


def load_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return [], "json_array"

    if stripped[0] == "[":
        data = json.loads(stripped)
        if not isinstance(data, list):
            raise SystemExit("JSON array input must contain a list of records")
        return [validate_record(item, index + 1) for index, item in enumerate(data)], "json_array"

    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        records.append(validate_record(json.loads(line), line_no))
    return records, "jsonl"


def validate_record(item: Any, line_no: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise SystemExit(f"record {line_no} is not a JSON object")
    return item


def write_records(path: Path, records: list[dict[str, Any]], input_format: str) -> None:
    with path.open("w", encoding="utf-8") as file:
        if input_format == "json_array":
            json.dump(records, file, ensure_ascii=False, indent=2)
            file.write("\n")
        else:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def pop_first(mapping: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping.pop(name)
    return default


def parse_torch_dtype(value: Any):
    if value is None:
        return None
    if not isinstance(value, str):
        return value

    import torch

    normalized = value.lower()
    aliases = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in aliases:
        raise SystemExit(f"unsupported torch_dtype: {value}")
    return aliases[normalized]


def normalize_model_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(kwargs)
    if "torch_dtype" in kwargs:
        kwargs["torch_dtype"] = parse_torch_dtype(kwargs["torch_dtype"])
    return kwargs


def get_auto_model_class(transformers_module, auto_class: str):
    mapping = {
        "causal_lm": "AutoModelForCausalLM",
        "seq2seq_lm": "AutoModelForSeq2SeqLM",
        "vision2seq": "AutoModelForVision2Seq",
    }
    class_name = mapping.get(auto_class, auto_class)
    if not hasattr(transformers_module, class_name):
        raise SystemExit(f"transformers does not provide model class {class_name!r}")
    return getattr(transformers_module, class_name)


def load_tokenizer_and_model(config: dict[str, Any]):
    import torch
    import transformers
    from transformers import AutoTokenizer

    model_cfg = dict(config.get("model", {}))
    tokenizer_cfg = dict(config.get("tokenizer", {}))

    model_path = pop_first(model_cfg, ["model_name_or_path", "model_path", "model"])
    if not model_path:
        raise SystemExit("config.model.model_name_or_path is required")

    tokenizer_path = pop_first(
        tokenizer_cfg,
        ["tokenizer_name_or_path", "tokenizer_path", "tokenizer"],
        model_cfg.pop("tokenizer_name_or_path", model_path),
    )
    adapter_path = pop_first(model_cfg, ["adapter_name_or_path", "adapter_path", "lora_adapter", "lora"])
    adapter_kwargs = model_cfg.pop("adapter_kwargs", {}) or {}
    merge_adapter = bool(model_cfg.pop("merge_adapter", False))
    auto_class = model_cfg.pop("auto_class", "causal_lm")

    tokenizer_cfg.setdefault("trust_remote_code", model_cfg.get("trust_remote_code", True))
    model_cfg.setdefault("trust_remote_code", True)
    if torch.cuda.is_available() and "device_map" not in model_cfg:
        model_cfg["device_map"] = "auto"

    logging.info("Loading tokenizer: %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_cfg)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "left"

    model_kwargs = normalize_model_kwargs(model_cfg)
    model_class = get_auto_model_class(transformers, auto_class)
    logging.info("Loading model: %s", model_path)
    model = model_class.from_pretrained(model_path, **model_kwargs)

    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("PEFT is required to load LoRA adapters: pip install peft") from exc

        adapter_paths = adapter_path if isinstance(adapter_path, list) else [adapter_path]
        for one_adapter in adapter_paths:
            logging.info("Loading LoRA adapter: %s", one_adapter)
            model = PeftModel.from_pretrained(model, one_adapter, **adapter_kwargs)
        if merge_adapter:
            logging.info("Merging LoRA adapter into base model")
            model = model.merge_and_unload()

    model.eval()
    return tokenizer, model


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def render_history(history: Any) -> str:
    if not history:
        return ""
    lines: list[str] = []
    if isinstance(history, list):
        for item in history:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                lines.append(f"user: {as_text(item[0])}")
                lines.append(f"assistant: {as_text(item[1])}")
            elif isinstance(item, dict):
                role = as_text(item.get("role", ""))
                content = as_text(item.get("content", ""))
                if role or content:
                    lines.append(f"{role}: {content}".strip(": "))
    return "\n".join(lines)


def build_messages(record: dict[str, Any], runtime_cfg: dict[str, Any]) -> list[dict[str, str]]:
    system_field = runtime_cfg.get("system_field", "system")
    instruction_field = runtime_cfg.get("instruction_field", "instruction")
    input_field = runtime_cfg.get("input_field", "input")
    history_field = runtime_cfg.get("history_field", "history")

    messages: list[dict[str, str]] = []
    system = as_text(record.get(system_field, "")).strip()
    if system:
        messages.append({"role": "system", "content": system})

    history = record.get(history_field, [])
    if isinstance(history, list):
        for item in history:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                messages.append({"role": "user", "content": as_text(item[0])})
                messages.append({"role": "assistant", "content": as_text(item[1])})
            elif isinstance(item, dict) and item.get("role") and item.get("content") is not None:
                messages.append({"role": as_text(item["role"]), "content": as_text(item["content"])})

    instruction = as_text(record.get(instruction_field, "")).strip()
    input_text = as_text(record.get(input_field, "")).strip()
    user_text = "\n".join(part for part in [instruction, input_text] if part)
    messages.append({"role": "user", "content": user_text})
    return messages


def build_prompt(record: dict[str, Any], tokenizer: Any, runtime_cfg: dict[str, Any]) -> str:
    prompt_template = runtime_cfg.get("prompt_template")
    if prompt_template:
        return prompt_template.format(
            system=as_text(record.get(runtime_cfg.get("system_field", "system"), "")),
            instruction=as_text(record.get(runtime_cfg.get("instruction_field", "instruction"), "")),
            input=as_text(record.get(runtime_cfg.get("input_field", "input"), "")),
            history=render_history(record.get(runtime_cfg.get("history_field", "history"), [])),
        )

    messages = build_messages(record, runtime_cfg)
    use_chat_template = runtime_cfg.get("use_chat_template", True)
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=runtime_cfg.get("add_generation_prompt", True),
        )

    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def model_input_device(model: Any):
    import torch

    if hasattr(model, "device") and str(model.device) != "meta":
        return model.device
    try:
        for parameter in model.parameters():
            if parameter.device != torch.device("meta"):
                return parameter.device
    except Exception:
        pass
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def cuda_synchronize_if_needed() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def generation_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    generation_cfg = dict(config.get("generation", {}))
    if generation_cfg.get("temperature") == 0:
        generation_cfg["do_sample"] = False
        generation_cfg.pop("temperature", None)
    return generation_cfg


def run_inference(
    records: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    config: dict[str, Any],
    prediction_field: str,
) -> tuple[list[dict[str, Any]], float]:
    runtime_cfg = dict(config.get("runtime", {}))
    batch_size = int(runtime_cfg.get("batch_size", 1))
    max_input_length = runtime_cfg.get("max_input_length")
    strip_prediction = runtime_cfg.get("strip_prediction", True)
    skip_special_tokens = runtime_cfg.get("skip_special_tokens", True)
    generation_kwargs = generation_kwargs_from_config(config)
    device = model_input_device(model)
    output_records: list[dict[str, Any]] = []
    total_inference_seconds = 0.0

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        prompts = [build_prompt(record, tokenizer, runtime_cfg) for record in batch_records]
        logging.info("Infer batch %d-%d / %d", start + 1, start + len(batch_records), len(records))

        cuda_synchronize_if_needed()
        batch_start = time.perf_counter()
        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=max_input_length is not None,
            max_length=max_input_length,
        )
        tokenized = move_batch_to_device(tokenized, device)
        input_width = tokenized["input_ids"].shape[1]

        import torch

        with torch.inference_mode():
            generated = model.generate(**tokenized, **generation_kwargs)
        decoded = tokenizer.batch_decode(generated[:, input_width:], skip_special_tokens=skip_special_tokens)
        if strip_prediction:
            decoded = [text.strip() for text in decoded]
        cuda_synchronize_if_needed()
        total_inference_seconds += time.perf_counter() - batch_start

        for record, prediction in zip(batch_records, decoded):
            item = copy.deepcopy(record)
            item[prediction_field] = prediction
            output_records.append(item)

    return output_records, total_inference_seconds


def normalize_text_for_metrics(text: str, metric_cfg: dict[str, Any]) -> str:
    normalized = text.strip() if metric_cfg.get("strip", True) else text
    if metric_cfg.get("lowercase", True):
        normalized = normalized.lower()
    if metric_cfg.get("remove_punctuation", False):
        normalized = re.sub(r"[^\w\s]", " ", normalized)
    if metric_cfg.get("collapse_spaces", True):
        normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def words_for_metrics(text: str, metric_cfg: dict[str, Any]) -> list[str]:
    normalized = normalize_text_for_metrics(text, metric_cfg)
    return normalized.split() if normalized else []


def edit_counts(ref_words: list[str], hyp_words: list[str]) -> tuple[int, int, int, int]:
    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1
    dp: list[list[tuple[int, int, int, int]]] = [[(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)]

    for i in range(1, rows):
        cost, sub, delete, insert = dp[i - 1][0]
        dp[i][0] = (cost + 1, sub, delete + 1, insert)
    for j in range(1, cols):
        cost, sub, delete, insert = dp[0][j - 1]
        dp[0][j] = (cost + 1, sub, delete, insert + 1)

    for i in range(1, rows):
        for j in range(1, cols):
            if ref_words[i - 1] == hyp_words[j - 1]:
                candidates = [dp[i - 1][j - 1]]
            else:
                cost, sub, delete, insert = dp[i - 1][j - 1]
                candidates = [(cost + 1, sub + 1, delete, insert)]

            cost, sub, delete, insert = dp[i - 1][j]
            candidates.append((cost + 1, sub, delete + 1, insert))
            cost, sub, delete, insert = dp[i][j - 1]
            candidates.append((cost + 1, sub, delete, insert + 1))
            dp[i][j] = min(candidates, key=lambda item: item[0])

    return dp[-1][-1]


def compute_metrics(records: list[dict[str, Any]], config: dict[str, Any], prediction_field: str, inference_seconds: float | None) -> dict[str, Any]:
    runtime_cfg = dict(config.get("runtime", {}))
    metric_cfg = dict(config.get("metrics", {}))
    reference_field = runtime_cfg.get("reference_field", "output")
    duration_field = runtime_cfg.get("duration_field", "duration")

    total_ref_words = 0
    substitutions = 0
    deletions = 0
    insertions = 0
    wrong_cases: list[dict[str, Any]] = []
    total_duration = 0.0

    for index, record in enumerate(records, 1):
        ref = as_text(record.get(reference_field, ""))
        hyp = as_text(record.get(prediction_field, ""))
        ref_words = words_for_metrics(ref, metric_cfg)
        hyp_words = words_for_metrics(hyp, metric_cfg)
        _, sub, delete, insert = edit_counts(ref_words, hyp_words)
        substitutions += sub
        deletions += delete
        insertions += insert
        total_ref_words += len(ref_words)

        if normalize_text_for_metrics(ref, metric_cfg) != normalize_text_for_metrics(hyp, metric_cfg):
            wrong_cases.append({"index": index, "ref": ref, "hyp": hyp})

        duration = record.get(duration_field)
        if isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0:
            total_duration += float(duration)

    sentence_count = len(records)
    wer = (substitutions + deletions + insertions) / total_ref_words if total_ref_words else 0.0
    ser = len(wrong_cases) / sentence_count if sentence_count else 0.0
    rtf = inference_seconds / total_duration if inference_seconds is not None and total_duration > 0 else None
    avg_latency = inference_seconds / sentence_count if inference_seconds is not None and sentence_count else None
    samples_per_second = sentence_count / inference_seconds if inference_seconds and inference_seconds > 0 else None
    return {
        "sentence_number": sentence_count,
        "ref_word_number": total_ref_words,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "wer": wer,
        "ser": ser,
        "wrong_sentence_number": len(wrong_cases),
        "inference_seconds": inference_seconds,
        "inference_rtf": rtf,
        "duration_seconds": total_duration if total_duration > 0 else None,
        "avg_latency_seconds": avg_latency,
        "samples_per_second": samples_per_second,
        "wrong_cases": wrong_cases,
    }


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    rtf = metrics["inference_rtf"]
    rtf_text = f"{rtf:.6f}" if rtf is not None else "N/A (no positive duration field found)"
    inference_seconds = metrics["inference_seconds"]
    inference_seconds_text = f"{inference_seconds:.6f}" if inference_seconds is not None else "N/A"
    avg_latency = metrics["avg_latency_seconds"]
    avg_latency_text = f"{avg_latency:.6f}" if avg_latency is not None else "N/A"
    samples_per_second = metrics["samples_per_second"]
    samples_per_second_text = f"{samples_per_second:.6f}" if samples_per_second is not None else "N/A"
    lines = [
        f"sentence number: {metrics['sentence_number']}",
        f"ref word number: {metrics['ref_word_number']}",
        f"wer: {metrics['wer']:.6f} ({metrics['wer'] * 100:.2f}%)",
        f"ser: {metrics['ser']:.6f} ({metrics['ser'] * 100:.2f}%)",
        f"inference rtf: {rtf_text}",
        f"inference seconds: {inference_seconds_text}",
        f"avg latency seconds: {avg_latency_text}",
        f"samples per second: {samples_per_second_text}",
        "",
    ]
    for case in metrics["wrong_cases"]:
        lines.append(f"ref: {case['ref']}")
        lines.append(f"hyp: {case['hyp']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference on LLaMA-Factory-style data.")
    parser.add_argument("--config", required=True, help="YAML config with model, tokenizer, generation, runtime, and metric settings.")
    parser.add_argument("--input", required=True, help="Input JSON array or JSONL file.")
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output", help="Prediction output file. Uses JSON array or JSONL format to match input.")
    output_group.add_argument("--output-dir", help="Directory for predictions, inference_summary.json, and inference.log.")
    parser.add_argument("--prediction-field", help=f"Prediction field to add to each record. Defaults to {DEFAULT_PREDICTION_FIELD!r}.")
    parser.add_argument("--limit", type=positive_int, help="Only run the first N records.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    config = load_yaml(config_path)
    prediction_field = args.prediction_field or config.get("runtime", {}).get("prediction_field", DEFAULT_PREDICTION_FIELD)
    records, input_format = load_records(input_path)
    if args.limit:
        records = records[: args.limit]

    if args.output:
        prediction_path = Path(args.output).expanduser().resolve()
        output_root = prediction_path.parent
    else:
        assert output_dir is not None
        output_root = output_dir
        prediction_path = output_root / ("predictions.json" if input_format == "json_array" else "predictions.jsonl")
    output_root.mkdir(parents=True, exist_ok=True)
    setup_logging(output_root / "inference.log")

    logging.info("Config: %s", config_path)
    logging.info("Input: %s", input_path)
    logging.info("Prediction output: %s", prediction_path)
    logging.info("Records: %d", len(records))
    logging.info("Input format: %s", input_format)
    logging.info("Prediction field: %s", prediction_field)

    tokenizer, model = load_tokenizer_and_model(config)
    predicted_records, inference_seconds = run_inference(records, tokenizer, model, config, prediction_field)
    write_records(prediction_path, predicted_records, input_format)

    summary_path = output_root / "inference_summary.json"
    summary = {
        "input": str(input_path),
        "predictions": str(prediction_path),
        "record_count": len(predicted_records),
        "prediction_field": prediction_field,
        "input_format": input_format,
        "inference_seconds": inference_seconds,
        "avg_latency_seconds": inference_seconds / len(predicted_records) if predicted_records else None,
        "samples_per_second": len(predicted_records) / inference_seconds if inference_seconds > 0 else None,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    logging.info("Predictions: %s", prediction_path)
    logging.info("Inference summary: %s", summary_path)
    logging.info("Inference seconds: %.6f", inference_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
