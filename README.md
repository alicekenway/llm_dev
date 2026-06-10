# LLaMA-Factory Slurm Training And Evaluation Tools

This directory contains helper tools for running LLaMA-Factory training on Slurm clusters and evaluating the trained model on JSON/JSONL test sets.

## Directory Layout

- `slurm_llamafactory/`
  - Slurm launcher for LLaMA-Factory training.
  - Supports `sbatch --wait`, existing `salloc` allocations via `--jobid`, single GPU, single-node multi-GPU, and multi-node multi-GPU jobs.
  - Supports ZeRO stage selection with existing DeepSpeed config files.
- `llamafactory_eval_tool/`
  - Inference and statistics tools for post-training evaluation.
  - Inference and stats can run independently, or sequentially through the wrapper script.
- `llamafactory_export_tool/`
  - Exports a LLaMA-Factory LoRA adapter into a merged Hugging Face model for evaluation or serving.
- `testing/`
  - Scratch area for local tests or examples.

See each subdirectory README for full argument details.

## Training

Run the launcher directly when you want it to submit an `sbatch` job and wait for completion:

```bash
python3 slurm_llamafactory/llamafactory_slurm_launcher.py \
  --config /path/to/sft.yaml \
  --env-dir /path/to/llamafactory_env \
  --deepspeed-dir /path/to/LlamaFactory/examples/deepspeed \
  --partition h200.141gb \
  --job-name qwen_lora_train \
  --nodes 1 \
  --gpus-per-node 1 \
  --zero-stage 0 \
  --time 24:00:00 \
  --mem 120G \
  --log-dir /path/to/logs
```

Use a pre-allocated `salloc` job by passing the allocation id:

```bash
python3 slurm_llamafactory/llamafactory_slurm_launcher.py \
  --config /path/to/sft.yaml \
  --env-dir /path/to/llamafactory_env \
  --deepspeed-dir /path/to/LlamaFactory/examples/deepspeed \
  --jobid 12345678 \
  --zero-stage 3 \
  --log-dir /path/to/logs
```

Important training notes:

- Do not wrap the normal `sbatch` launcher command with `srun`. The launcher submits and waits for the training job itself.
- Use `srun` only when running inside a pre-existing allocation workflow if your cluster requires it, and prefer the launcher's `--jobid` mode for that case.
- `--deepspeed-dir` is the input directory containing existing DeepSpeed config files. It is not an output directory.
- Keep config files, logs, datasets, model paths, and output paths on filesystems visible to all allocated nodes.

## Evaluation

The evaluation tool expects records like the training set:

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

Run inference and statistics in one command:

```bash
llamafactory_eval_tool/run_eval.sh both \
  --config llamafactory_eval_tool/example_config.yaml \
  --env-dir /path/to/llamafactory_env \
  --input /path/to/test.json \
  --output-dir /path/to/eval_out
```

Run multiple test sets by passing an input-list JSON:

```json
{
  "dev": "/path/to/dev.json",
  "test": "/path/to/test.jsonl"
}
```

```bash
llamafactory_eval_tool/run_eval.sh both \
  --config llamafactory_eval_tool/example_config.yaml \
  --env-dir /path/to/llamafactory_env \
  --input_list /path/to/input_list.json \
  --output-dir /path/to/eval_out
```

Or run them separately:

```bash
llamafactory_eval_tool/run_eval.sh infer \
  --config llamafactory_eval_tool/example_config.yaml \
  --env-dir /path/to/llamafactory_env \
  --input /path/to/test.json \
  --output-dir /path/to/eval_out

llamafactory_eval_tool/run_eval.sh stats \
  --config llamafactory_eval_tool/example_config.yaml \
  --input /path/to/eval_out/predictions.json \
  --output-dir /path/to/eval_out
```

Evaluation outputs:

- `predictions.json` or `predictions.jsonl`: original records plus a `prediction` field.
- `inference.log`: runtime log from inference.
- `inference_summary.json`: elapsed time, item count, and throughput metadata.
- `report.txt`: summary header with sentence count, reference word count, WER, CER, SER, optional inference RTF, followed by wrong cases.
- `metrics.json`: machine-readable statistics.
- Batch mode writes per-set predictions under `res/`, per-set reports and metrics under `stat/`, and a top-level `summary.tsv` table with test set name, sentence count, word count, WER, CER, and RTF.

Stats can be rerun without loading the model as long as the prediction file already exists.
Prediction files are flushed after each inference batch, so you can inspect partial results while a set is still running.
For Qwen-style thinking models, set `runtime.enable_thinking: false` in the eval YAML to prevent reasoning text from appearing in predictions.
Metrics are case-insensitive and ignore punctuation by default.

For evaluation, pass `--env-dir /path/to/env` to source `/path/to/env/bin/activate` and run the Python tools with that environment's interpreter.

## Common Workflow

1. Prepare the LLaMA-Factory training YAML.
2. Launch training with `slurm_llamafactory/llamafactory_slurm_launcher.py`.
3. If direct adapter loading is not compatible with your eval or serving stack, merge the adapter with `llamafactory_export_tool/export_merged_model.py`.
4. Put the trained adapter or merged model path into an evaluation YAML.
5. Run `llamafactory_eval_tool/run_eval.sh both`.
6. Inspect `report.txt` for aggregate quality and wrong cases.

## More Documentation

- [Training launcher README](slurm_llamafactory/README.md)
- [Merged model exporter README](llamafactory_export_tool/README.md)
- [Evaluation tool README](llamafactory_eval_tool/README.md)
