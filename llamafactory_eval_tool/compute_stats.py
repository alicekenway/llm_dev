#!/usr/bin/env python3
"""Compute WER/CER/SER report from prediction records.

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
    load_input_list,
    load_records,
    load_yaml,
    write_report,
)


def read_inference_seconds(summary_path: Path | None, test_set_name: str | None = None) -> float | None:
    if summary_path is None or not summary_path.exists():
        return None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if test_set_name is not None and isinstance(data.get("sets"), list):
        for item in data["sets"]:
            if isinstance(item, dict) and item.get("name") == test_set_name:
                value = item.get("inference_seconds")
                return float(value) if isinstance(value, (int, float)) else None
    value = data.get("inference_seconds")
    return float(value) if isinstance(value, (int, float)) else None


def format_optional_float(value: Any) -> str:
    return f"{value:.6f}" if isinstance(value, (int, float)) else "N/A"


def write_summary_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["test_set_name\tsentence_num\tword_num\twer\tcer\trtf"]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["name"]),
                    str(row["sentence_number"]),
                    str(row["ref_word_number"]),
                    f"{row['wer']:.6f}",
                    f"{row['cer']:.6f}",
                    format_optional_float(row["inference_rtf"]),
                ]
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute WER/CER/SER statistics from prediction records.")
    parser.add_argument("--config", help="YAML config. Only runtime and metrics sections are used.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Prediction JSON array or JSONL file containing reference and prediction fields.")
    input_group.add_argument("--input-list", "--input_list", help="JSON object mapping test set names to prediction files.")
    parser.add_argument("--output-dir", help="Directory for report.txt and metrics.json.")
    parser.add_argument("--report", help="Report text path. Defaults to <output-dir>/report.txt or input directory.")
    parser.add_argument("--metrics-json", help="Metrics JSON path. Defaults to <output-dir>/metrics.json or input directory.")
    parser.add_argument("--summary", help="Batch summary table path. Defaults to <output-dir>/summary.tsv.")
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
    config: dict[str, Any] = load_yaml(Path(args.config).expanduser().resolve()) if args.config else {}
    prediction_field = args.prediction_field or config.get("runtime", {}).get("prediction_field", DEFAULT_PREDICTION_FIELD)

    if args.input_list:
        if args.report or args.metrics_json:
            raise SystemExit("--report and --metrics-json are only supported with single --input")
        input_list_path = Path(args.input_list).expanduser().resolve()
        prediction_items = load_input_list(input_list_path)
        output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_list_path.parent
        stat_dir = output_root / "stat"
        stat_dir.mkdir(parents=True, exist_ok=True)
        summary_path = Path(args.summary).expanduser().resolve() if args.summary else output_root / "summary.tsv"
        batch_summary_path = Path(args.inference_summary).expanduser().resolve() if args.inference_summary else None

        rows: list[dict[str, Any]] = []
        for name, prediction_path in prediction_items:
            records, _ = load_records(prediction_path)
            if args.inference_seconds is not None:
                inference_seconds = args.inference_seconds
            else:
                inference_seconds = read_inference_seconds(batch_summary_path, name)
                if inference_seconds is None:
                    inference_seconds = read_inference_seconds(prediction_path.parent / f"{name}_summary.json")

            metrics = compute_metrics(records, config, prediction_field, inference_seconds)
            report_path = stat_dir / f"{name}_stat.txt"
            metrics_path = stat_dir / f"{name}_stat.json"
            write_report(report_path, metrics)
            metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            rows.append(
                {
                    "name": name,
                    "sentence_number": metrics["sentence_number"],
                    "ref_word_number": metrics["ref_word_number"],
                    "wer": metrics["wer"],
                    "cer": metrics["cer"],
                    "inference_rtf": metrics["inference_rtf"],
                }
            )

        write_summary_table(summary_path, rows)
        print(f"Summary: {summary_path}")
        print(f"Stat dir: {stat_dir}")
        return 0

    assert args.input is not None
    input_path = Path(args.input).expanduser().resolve()

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
    print(f"CER: {metrics['cer']:.6f}")
    print(f"SER: {metrics['ser']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
