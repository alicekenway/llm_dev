# LLaMA-Factory Evaluation Tool

This directory contains separate inference and statistics tools for LLaMA-Factory-style JSON or JSONL data.

- `run_inference.py`: loads the model and writes predictions.
- `compute_stats.py`: reads prediction records and computes WER/CER/SER.
- `run_eval.sh`: wrapper that can run `infer`, `stats`, or `both`.

Input records can look like:

```json
[
  {
    "instruction": "",
    "input": "the music one",
    "output": "increase music volume",
    "system": "You are an intelligent in-car voice assistant rewrite engine...",
    "history": [["make it louder", "increase volume"]]
  }
]
```

## Run Both

```bash
llamafactory_eval_tool/run_eval.sh both \
  --config llamafactory_eval_tool/example_config.yaml \
  --env-dir /path/to/llamafactory_env \
  --input /path/to/test.json \
  --output-dir /path/to/eval_out
```

Outputs:

- `predictions.json` or `predictions.jsonl`: same format as input, with a `prediction` field added to each record
- `inference_summary.json`: inference time and prediction metadata
- `inference.log`: inference log
- `report.txt`: statistics header and wrong cases
- `metrics.json`: machine-readable metrics

Prediction files are streamed to disk after each inference batch, so partial results are visible before the full set finishes.

## Run Separately

Inference only:

```bash
python3 llamafactory_eval_tool/run_inference.py \
  --config llamafactory_eval_tool/example_config.yaml \
  --input /path/to/test.json \
  --output-dir /path/to/eval_out
```

Statistics only:

```bash
python3 llamafactory_eval_tool/compute_stats.py \
  --input /path/to/eval_out/predictions.json \
  --output-dir /path/to/eval_out \
  --inference-summary /path/to/eval_out/inference_summary.json
```

Add `--config llamafactory_eval_tool/example_config.yaml` when you need custom field names or metric normalization.

The wrapper can run either part:

```bash
llamafactory_eval_tool/run_eval.sh infer \
  --config llamafactory_eval_tool/example_config.yaml \
  --env-dir /path/to/llamafactory_env \
  --input /path/to/test.json \
  --output-dir /path/to/eval_out

llamafactory_eval_tool/run_eval.sh stats \
  --input /path/to/eval_out/predictions.json \
  --output-dir /path/to/eval_out
```

Pass `--env-dir /path/to/env` when you want the wrapper to source `/path/to/env/bin/activate` and run the tools with that environment's Python. `--env-path` is accepted as an alias.

You can also write predictions to an explicit file:

```bash
python3 llamafactory_eval_tool/run_inference.py \
  --config llamafactory_eval_tool/example_config.yaml \
  --input /path/to/test.jsonl \
  --output /path/to/predictions.jsonl
```

## Config

The YAML has open-ended sections:

- `model`: passed to `AutoModelForCausalLM.from_pretrained` after reserved keys are removed
- `tokenizer`: passed to `AutoTokenizer.from_pretrained`
- `generation`: passed to `model.generate`
- `runtime`: field names, batch size, prompt behavior, max input length
- `metrics`: text normalization for WER/SER

Reserved `model` keys:

- `model_name_or_path`
- `adapter_name_or_path`
- `adapter_kwargs`
- `merge_adapter`
- `auto_class`: `causal_lm`, `seq2seq_lm`, `vision2seq`, or a Transformers auto class name

Example:

```yaml
model:
  model_name_or_path: /path/to/base_model
  adapter_name_or_path: /path/to/lora_adapter
  trust_remote_code: true
  torch_dtype: bfloat16
  device_map: auto

generation:
  max_new_tokens: 128
  do_sample: false
  repetition_penalty: 1.05

runtime:
  batch_size: 4
  max_input_length: 4096
  prediction_field: prediction
  enable_thinking: false
  remove_thinking: true

metrics:
  lowercase: true
  remove_punctuation: true
  collapse_spaces: true
```

For Qwen-style thinking models, keep `runtime.enable_thinking: false`. Otherwise the chat template may allow the model to generate reasoning text such as `Thinking Process:` before the final rewrite. `runtime.remove_thinking: true` also strips explicit `<think>...</think>` blocks if the model still emits them.

Metrics are case-insensitive by default and ignore punctuation by default. This affects WER, CER, SER, and wrong-case detection.

## Report Format

The report header contains:

```text
sentence number: 100
ref word number: 432
ref char number: 2140
wer: 0.034722 (3.47%)
cer: 0.012150 (1.21%)
ser: 0.120000 (12.00%)
inference rtf: N/A (no positive duration field found)
```

Below the header, every wrong case is listed as:

```text
ref: increase music volume
hyp: raise music volume
```

RTF is computed only when records contain a positive numeric duration field, default `duration`. For text-only data without duration, the report prints RTF as `N/A` and still reports inference seconds, average latency, and samples per second.

## Multiple Test Sets

Single-file `--input` still works. To run multiple test sets sequentially, create an input-list JSON object:

```json
{
  "dev": "/path/to/dev.json",
  "test": "/path/to/test.jsonl"
}
```

Then run:

```bash
llamafactory_eval_tool/run_eval.sh both \
  --config llamafactory_eval_tool/example_config.yaml \
  --env-dir /path/to/llamafactory_env \
  --input_list /path/to/input_list.json \
  --output-dir /path/to/eval_out
```

Batch output layout:

```text
eval_out/
  res/
    dev_res.json
    dev_summary.json
    test_res.jsonl
    test_summary.json
  stat/
    dev_stat.txt
    dev_stat.json
    test_stat.txt
    test_stat.json
  batch_inference_summary.json
  prediction_list.json
  summary.tsv
```

`summary.tsv` contains:

```text
test_set_name	sentence_num	word_num	wer	cer	rtf
```

Batch modes can also run independently:

```bash
llamafactory_eval_tool/run_eval.sh infer \
  --config llamafactory_eval_tool/example_config.yaml \
  --input-list /path/to/input_list.json \
  --output-dir /path/to/eval_out

llamafactory_eval_tool/run_eval.sh stats \
  --input-list /path/to/eval_out/prediction_list.json \
  --output-dir /path/to/eval_out \
  --inference-summary /path/to/eval_out/batch_inference_summary.json
```

`--input-list` and `--input_list` are both accepted. In `stats` mode, the list should point to prediction files, not raw input files.
