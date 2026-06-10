# LLaMA-Factory Merged Model Export Tool

This tool exports a LLaMA-Factory LoRA adapter into a merged Hugging Face model directory.
Use it when direct PEFT adapter loading does not match the base model structure used by another eval or serving tool.

The wrapper generates a LLaMA-Factory export config and runs:

```bash
llamafactory-cli export <generated_config>
```

The original base model and adapter directories are not modified.

## Basic Usage

```bash
python3 llamafactory_export_tool/export_merged_model.py \
  --base-model /path/to/base_model \
  --adapter /path/to/lora_adapter \
  --output-dir /path/to/merged_model \
  --env-dir /path/to/llamafactory_env
```

By default, the generated config uses the classic `llamafactory-cli export` keys:

```yaml
model_name_or_path: /path/to/base_model
adapter_name_or_path: /path/to/lora_adapter
finetuning_type: lora
template: qwen3_nothink
export_dir: /path/to/merged_model
export_size: 5
export_device: auto
infer_dtype: auto
trust_remote_code: true
```

If `--template` is omitted, the wrapper searches the adapter directory and nearby parent directories for a training YAML such as `sft.yaml` and copies its `template` value.
Pass `--template TEMPLATE` to set it explicitly, or `--no-template-auto-detect` to leave it unset.

Run under Slurm when the model needs a GPU:

```bash
srun --partition=h200.141gb --gres=gpu:1 --mem=120GB \
  python3 llamafactory_export_tool/export_merged_model.py \
    --base-model /mnt/users/jinyang_wang/LLM/model/Qwen3.5-9B \
    --adapter /mnt/users/jinyang_wang/LLM/training_expts_llamaFactory/expts1/saves/qwen3-30b-a3b/lora/sft-code \
    --output-dir /mnt/users/jinyang_wang/LLM/training_expts_llamaFactory/exports/expts1_merged \
    --training-config /mnt/users/jinyang_wang/LLM/training_expts_llamaFactory/expts1/sft.yaml \
    --env-dir /mnt/users/jinyang_wang/LLM/llamafactory_env
```

Then evaluate the merged model without `adapter_name_or_path`:

```yaml
model:
  model_name_or_path: /path/to/merged_model
  trust_remote_code: true
  torch_dtype: bfloat16
  device_map: auto
  auto_class: causal_lm
```

## Options

- `--config-format classic`: default, writes flat LLaMA-Factory export keys such as `model_name_or_path`, `adapter_name_or_path`, and `export_dir`.
- `--config-format v1`: writes the newer `model` plus `peft_config` structure for LLaMA-Factory installs that use the v1 export path.
- `--template TEMPLATE`: explicit LLaMA-Factory template. For the current Qwen3.5 runs, use `qwen3_nothink`, matching the training YAML.
- `--training-config PATH`: read `template` from a specific training YAML when `--template` is omitted.
- `--no-template-auto-detect`: do not infer `template` from nearby training configs.
- `--infer-dtype auto|float16|bfloat16|float32`: dtype used during export.
- `--export-size N`: model shard size in GB.
- `--allow-existing-output`: allow writing into a non-empty output directory.
- `--set KEY=VALUE`: override generated config values, including dotted keys like `peft_config.infer_dtype=bfloat16`.
- `--dry-run`: write the generated config and command, but do not run export.

Generated configs and command metadata are written under `llamafactory_export_tool/runs/`.
