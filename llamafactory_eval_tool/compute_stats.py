#!/usr/bin/env python3
"""Compute WER/SER report from prediction records.

This script does not load the model. It expects records that already contain a
prediction field, normally produced by run_inference.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_inference import (
    DEFAULT_PREDICTION_FIELD,
    compute_metrics,
    load_records,
    load_yaml,
    write_report,
)


def read_inference_seconds(summary_path: Path | None) -> float | None:
    if summary_path is None or not summary_path.exists():
        return None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    value = data.get("inference_seconds")
    return float(value) if isinstance(value, (int, float)) else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute WER/SER statistics from prediction records.")
    parser.add_argument("--config", help="YAML config. Only runtime and metrics sections are used.")
    parser.add_argument("--input", required=True, help="Prediction JSON array or JSONL file containing reference and prediction fields.")
    parser.add_argument("--output-dir", help="Directory for report.txt and metrics.json.")
    parser.add_argument("--report", help="Report text path. Defaults to <output-dir>/report.txt or input directory.")
    parser.add_argument("--metrics-json", help="Metrics JSON path. Defaults to <output-dir>/metrics.json or input directory.")
    parser.add_argument("--prediction-field", help=f"Prediction field. Defaults to config runtime value or {DEFAULT_PREDICTION_FIELD!r}.")
    parser.add_argument("--inference-summary", help="Path to inference_summary.json from run_inference.py.")
    parser.add_argument("--inference-seconds", type=float, help="Override inference seconds for RTF/latency stats.")
    return parser


def resolve_output_paths(args: argparse.Namespace, input_path: Path) -> tuple[Path, Path]:
    if args.output_dir:
        output_root = Path(args.output_dir).expanduser().resolve()
    else:
        output_root = input_path.parent
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report).expanduser().resolve() if args.report else output_root / "report.txt"
    metrics_path = Path(args.metrics_json).expanduser().resolve() if args.metrics_json else output_root / "metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    return report_path, metrics_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    config: dict[str, Any] = load_yaml(Path(args.config).expanduser().resolve()) if args.config else {}
    prediction_field = args.prediction_field or config.get("runtime", {}).get("prediction_field", DEFAULT_PREDICTION_FIELD)

    records, _ = load_records(input_path)
    if args.inference_seconds is not None:
        inference_seconds = args.inference_seconds
    else:
        summary_path = Path(args.inference_summary).expanduser().resolve() if args.inference_summary else input_path.parent / "inference_summary.json"
        inference_seconds = read_inference_seconds(summary_path)

    metrics = compute_metrics(records, config, prediction_field, inference_seconds)
    report_path, metrics_path = resolve_output_paths(args, input_path)
    write_report(report_path, metrics)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Report: {report_path}")
    print(f"Metrics: {metrics_path}")
    print(f"WER: {metrics['wer']:.6f}")
    print(f"SER: {metrics['ser']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
