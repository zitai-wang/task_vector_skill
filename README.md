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

Important:

- this should produce a compatible environment, not a byte-for-byte clone of the author's machine
- CUDA, driver, OS, and local model installation still matter
- if you need a closer machine snapshot, you can distribute a `conda-pack` archive in addition to `environment.yml`

### Using the Packed `licv` Environment

If `artifacts/licv-conda-pack.tar.gz` has already been generated, it is a packed snapshot of the `licv` Conda environment.

It is not used by opening the archive directly. Instead, unpack it into a target directory and activate it from there:

```bash
mkdir -p /path/to/licv
tar -xzf artifacts/licv-conda-pack.tar.gz -C /path/to/licv
source /path/to/licv/bin/activate
conda-unpack
```

After that, verify the environment:

```bash
python -V
which python
```

Notes:

- the packed environment is best suited for similar Linux machines
- it does not guarantee compatibility across different operating systems, CUDA versions, GPU drivers, or hardware setups
- run `conda-unpack` once after the first extraction so paths inside the packed environment are fixed
- this archive is closer to an exact snapshot of the original machine than `environment.yml`
- for general sharing, keep `environment.yml` as the portable option and use the packed archive only when users need a closer runtime snapshot

## Required Local Resources

This repo expects local model weights and local dataset files.
Do not hardcode your own server paths into the code when sharing this repo.
Prefer environment-variable overrides.

Recommended overrides:

```bash
export COT_MIMIC_RUNNER_PYTHON=/path/to/python
export COT_MIMIC_CONDA_ACTIVATE='source /path/to/conda.sh && conda activate licv'
export COT_MIMIC_CACHE_DIR=/path/to/cache
export COT_MIMIC_RESULT_DIR=/path/to/results

export COT_MIMIC_MODEL_ROOT_QWEN2_5_7B_INSTRUCT=/path/to/Qwen2.5-7B-Instruct

export COT_MIMIC_DATASET_SOURCE_GSM8K=/path/to/gsm8k
export COT_MIMIC_DATASET_SOURCE_COMMONSENSEQA=/path/to/commonsenseqa
export COT_MIMIC_DATASET_SOURCE_STRATEGYQA=/path/to/strategyqa
```

The pipeline controller reads these variables directly.
If required resources are missing, it now fails with an actionable message instead of a deep traceback.

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

## Notes for Sharing

If you want other people to use this repo successfully:

- share `environment.yml`
- share the required local model and dataset paths, or document the matching `COT_MIMIC_*` variables
- avoid assuming `/home/...`, `/data/share/...`, or your own conda path exists on their machine
- if you distribute a packed environment, document that it is best for similar Linux machines and CUDA setups

## Current Default Recommendation

If someone is new to this repo, start here:

1. create `licv` from `environment.yml`
2. set the `COT_MIMIC_MODEL_ROOT_*` and `COT_MIMIC_DATASET_SOURCE_*` variables
3. run `scripts/run_static_pipeline.py`
4. use the pipeline skill if working through Codex