# Task vector skill

`Task vector skill` is a research codebase for static Chain-of-Thought transfer and evaluation.

The current default workflow in this repo is:

- use a source LLM to generate self-CoT
- extract static CoT vectors from correct samples
- inject those vectors back into the same or another LLM
- evaluate `baseline`, `base`, `ffn`, or `attn` behavior

The most stable path in this repo today is the unimodal Qwen pipeline:

- model: `qwen2.5-7b-instruct`
- datasets: `gsm8k`, `commonsenseqa`, `strategyqa`
- workflow: `baseline -> generate -> extract -> eval`

This repo also contains earlier multimodal and training-oriented code, but the shared demo path we currently recommend is the static pipeline above.

## What This Repo Supports

Core capabilities:

- self-CoT generation with [src/generate_self_cot.py](/data1/wzy/cot-mimic%20copy/src/generate_self_cot.py)
- static vector extraction with [src/extract_cot_vector.py](/data1/wzy/cot-mimic%20copy/src/extract_cot_vector.py)
- baseline evaluation with [src/eval_baseline.py](/data1/wzy/cot-mimic%20copy/src/eval_baseline.py)
- base-vector evaluation with [src/eval_base.py](/data1/wzy/cot-mimic%20copy/src/eval_base.py)
- FFN / LIVE-style evaluation with [src/eval_licv.py](/data1/wzy/cot-mimic%20copy/src/eval_licv.py)
- attention / MimIC evaluation with [src/eval_mimic.py](/data1/wzy/cot-mimic%20copy/src/eval_mimic.py)
- end-to-end automation with [scripts/run_static_pipeline.py](/data1/wzy/cot-mimic%20copy/scripts/run_static_pipeline.py)
- layerwise sweeps with [scripts/run_eval_layers_base.py](/data1/wzy/cot-mimic%20copy/scripts/run_eval_layers_base.py), [scripts/run_eval_layers_licv.py](/data1/wzy/cot-mimic%20copy/scripts/run_eval_layers_licv.py), and [scripts/run_eval_layers_mimic.py](/data1/wzy/cot-mimic%20copy/scripts/run_eval_layers_mimic.py)

## Recommended Environment

The shared environment file is [environment.yml](/data1/wzy/cot-mimic%20copy/environment.yml).
It creates an environment named `licv`.

```bash
conda env create -f environment.yml
conda activate licv
```

What `environment.yml` is intended to cover:

- the default `licv` Conda environment name
- the core runtime for the shared static pipeline
- PyTorch `2.5.1` with CUDA `12.1`
- the main Hugging Face stack such as `transformers`, `datasets`, `accelerate`, and `peft`
- config and evaluation packages such as `hydra-core`, `omegaconf`, and `evaluate`
- repo-specific utility dependencies such as `deepspeed`, `sentencepiece`, `safetensors`, `bytecode`, `xxhash`, and `python-Levenshtein`


## Required Local Resources

The current shared default setup in this repo only supports:

- model: `qwen2.5-7b-instruct`
- datasets: `strategyqa`, `commonsenseqa`, `gsm8k`

This means anyone using the shared static pipeline should prepare:

- a local path to the `qwen2.5-7b-instruct` model weights
- a local path to the `strategyqa` dataset
- a local path to the `commonsenseqa` dataset
- a local path to the `gsm8k` dataset

These local paths are machine-specific.
When sharing this repo, do not keep your own server paths hardcoded in commands or configs.
Instead, replace the example paths below with the actual paths on your own machine.


## Method Guide

This repo uses four method names during evaluation:

- `baseline`: no extracted CoT vector is injected. This is the plain reference run.
- `base`: inject a static hidden-state style vector with [src/eval_base.py](/data1/wzy/cot-mimic%20copy/src/eval_base.py). In practice, this is the simplest "add the extracted vector back into the model" setting.
- `ffn`: inject the extracted vector through the FFN / MLP path with [src/eval_licv.py](/data1/wzy/cot-mimic%20copy/src/eval_licv.py). In this repo, this corresponds to the LIVE-style / LICV-style FFN shift path.
- `attn`: inject the extracted vector through the attention path with [src/eval_mimic.py](/data1/wzy/cot-mimic%20copy/src/eval_mimic.py). In this repo, this is the MimIC-style attention shift path.

## Quick Start

Run the default full StrategyQA attention pipeline on one visible GPU:

```bash
python scripts/run_static_pipeline.py \
  --model-name qwen2.5-7b-instruct \
  --dataset strategyqa \
  --method attn \
  --devices 0
```

What this does by default:

1. runs `baseline`
2. generates self-CoT on `500` training-side samples
3. extracts a run-local `.pt` vector
4. runs the requested eval method
5. for `base`, `ffn`, and `attn`, uses layerwise evaluation unless `--no-layerwise` is given

Useful variants:

```bash
# single eval instead of layerwise sweep
python scripts/run_static_pipeline.py \
  --model-name qwen2.5-7b-instruct \
  --dataset gsm8k \
  --method ffn \
  --devices 0 \
  --no-layerwise \
  --single-layer 15

# baseline only
python scripts/run_static_pipeline.py \
  --model-name qwen2.5-7b-instruct \
  --dataset commonsenseqa \
  --method baseline \
  --devices 0

# preview commands without executing
python scripts/run_static_pipeline.py \
  --model-name qwen2.5-7b-instruct \
  --dataset strategyqa \
  --method attn \
  --dry-run
```

## Outputs

Each pipeline run writes into:

```text
results/static_pipeline/<run_tag>/
```

Typical contents:

- `logs/`
- `records/`
- `self_cot_data.json`
- `self_cot_data_correct_only.json`
- `<run_tag>.pt`
- `run_summary.json`

## Skill Usage

The shared Codex skill for this workflow is:

- [skills/pipeline/SKILL.md](/data1/wzy/cot-mimic%20copy/skills/pipeline/SKILL.md)

Use this skill when you want Codex to:

- switch datasets between `gsm8k`, `commonsenseqa`, and `strategyqa`
- switch methods between `baseline`, `base`, `ffn`, and `attn`
- update Hydra overrides automatically
- run the full static pipeline end to end
- recover from small runtime/config errors during long jobs

What this skill does for you:

- chooses the matching eval entrypoint automatically
- keeps the recommended order `baseline -> generate -> extract -> eval`
- uses the method-matched layerwise runner for `base`, `ffn`, and `attn`
- updates `model_name`, `data.name`, `self_cot_path`, and `extracted_cot_vector_path` when the task is clear

How to use the skill with Codex:

- mention the pipeline skill and state the dataset, model, and method you want
- for full runs, ask for the whole pipeline from `baseline` to extraction and evaluation
- for analysis-only runs, ask for a single method or a single layer
- if you already have `self_cot_data_correct_only.json` or a `.pt` vector, say so and Codex can skip earlier stages

Example prompts for Codex:

```text
Use the pipeline skill to run the full static pipeline on strategyqa with qwen2.5-7b-instruct and method attn.
```

```text
Use the pipeline skill to run baseline and then an ffn layerwise sweep on gsm8k.
```

```text
Use the pipeline skill to evaluate the existing extracted vector at results/static_pipeline/my_run/my_run.pt with method base on layer 15 only.
```

```text
Use the pipeline skill to switch the current setup from commonsenseqa + ffn to strategyqa + attn and keep the same model.
```

A practical mapping from user intent to method:

- choose `baseline` when you want the no-vector reference
- choose `base` when you want the simplest direct vector injection baseline
- choose `ffn` when you want FFN / MLP-path injection
- choose `attn` when you want attention-path injection

## Project Layout

Main directories:

- [src](/data1/wzy/cot-mimic%20copy/src): core pipeline logic, eval entrypoints, vector extraction, model helpers
- [src/config](/data1/wzy/cot-mimic%20copy/src/config): Hydra config files
- [src/dataset_utils](/data1/wzy/cot-mimic%20copy/src/dataset_utils): dataset loaders and prompt formatting
- [scripts](/data1/wzy/cot-mimic%20copy/scripts): automation helpers and layerwise runners
- [skills](/data1/wzy/cot-mimic%20copy/skills): Codex skills for shared usage
- [results](/data1/wzy/cot-mimic%20copy/results): generated outputs and evaluation records

## Important Implementation Files

- [src/shift_encoder.py](/data1/wzy/cot-mimic%20copy/src/shift_encoder.py)

  This contains the main shift implementations, including MimIC-style attention shifts and FFN-based variants.

- [src/utils.py](/data1/wzy/cot-mimic%20copy/src/utils.py)

  This centralizes model resolution, runtime device selection, and several shared helpers used by the eval and extraction scripts.

- [scripts/run_static_pipeline.py](/data1/wzy/cot-mimic%20copy/scripts/run_static_pipeline.py)

  This is the recommended shared controller for static experiments. It handles stage ordering, logging, cache setup, dataset/model path overrides, and run summaries.

