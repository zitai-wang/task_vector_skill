---
name: multicot-static-pipeline
description: Run the cot-mimic static research pipeline end to end: generate self-CoT, extract CoT vectors, and evaluate them layerwise. Use this skill especially for the simple LLM to LLM setup with qwen2.5-7b-instruct on gsm8k, commonsenseqa, and strategyqa, including switching model_name, data.name, self_cot_path, extracted_cot_vector_path, and the matching eval yaml for baseline, base, ffn, or attn runs.
---

# MultiCoT Static Pipeline

This skill is for the full static workflow, not just extraction.

Default research setup:
- source model: `qwen2.5-7b-instruct`
- target model: `qwen2.5-7b-instruct`
- source dataset: `gsm8k`, `commonsenseqa`, or `strategyqa`
- target dataset: `gsm8k`, `commonsenseqa`, or `strategyqa`
- generate stage sample count: `500`
- eval stage sample count: full query set
- runtime python environment: `licv`
- shared environment file: `environment.yml`

Default pipeline:
1. run `baseline` eval first as the no-vector reference
2. generate self-CoT
3. extract vector from correct self-CoT samples
4. run the requested vector method such as `base`, `ffn`, or `attn`
5. if needed, run layerwise sweep

Important default:
- a "full pipeline" run should include `baseline`
- if the user does not explicitly disable it, `baseline` should be treated as part of the default end-to-end workflow
- for the default Qwen unimodal demos, the expected sequence is:
  - `baseline -> generate -> extract -> requested vector eval`

## When to use

Use this skill when the user wants to:
- run the full static pipeline
- generate self-CoT
- extract a `.pt` vector
- run baseline, base, ffn, or attn eval
- do layerwise eval with `run_eval_layers.py`
- switch between `gsm8k`, `commonsenseqa`, and `strategyqa`
- switch the model through `model_name`
- let Codex automatically edit the needed yaml parameters and run the workflow

Do not use this skill for:
- learnable recovery training
- VLM injection as the main path
- multimodal extraction as the default path

## Main idea

Keep the default story simple:
- use one LLM to extract
- use the same LLM to evaluate
- only demonstrate on unimodal datasets first

In this skill, the preferred path is:
- `LLM extract -> LLM eval`

More concretely:
- use `qwen2.5-7b-instruct`
- use `gsm8k`, `commonsenseqa`, or `strategyqa`
- first get `self_cot_data_correct_only.json`
- then get a run-local `.pt` vector inside the current run directory
- then evaluate with the right eval yaml
- by default, generate uses `500` samples
- by default, eval uses the full dataset

## Model and data paths

Portability rule:
- this skill is written for shared use, so do not rely on author-machine paths as the normal setup
- prefer environment-variable overrides before editing source files
- Codex should treat missing local paths as a configuration issue, not a research failure

Current shared support:
- model: `qwen2.5-7b-instruct`
- datasets: `gsm8k`, `commonsenseqa`, `strategyqa`

What users need to do after cloning this repo:
- prepare the `qwen2.5-7b-instruct` weights on their own machine
- prepare local copies of `gsm8k`, `commonsenseqa`, and `strategyqa` if they want to run those datasets
- replace every example `/path/to/...` value with a real local path on their machine
- use environment variables first instead of editing hardcoded paths into the repo

Recommended runtime overrides:
- `COT_MIMIC_RUNNER_PYTHON`
- `COT_MIMIC_CONDA_ACTIVATE`
- `COT_MIMIC_CACHE_DIR`
- `COT_MIMIC_RESULT_DIR`
- model-specific roots such as `COT_MIMIC_MODEL_ROOT_QWEN2_5_7B_INSTRUCT`
- dataset-specific roots such as `COT_MIMIC_DATASET_SOURCE_GSM8K`, `COT_MIMIC_DATASET_SOURCE_COMMONSENSEQA`, `COT_MIMIC_DATASET_SOURCE_STRATEGYQA`

Default shared setup:
- `model_name=qwen2.5-7b-instruct`
- runtime environment:
  - use the interpreter from `conda activate licv`
- model path:
  - should come from `COT_MIMIC_MODEL_ROOT_QWEN2_5_7B_INSTRUCT` on the user's machine
- dataset paths:
  - should come from the matching `COT_MIMIC_DATASET_SOURCE_*` variables on the user's machine

GSM8K:
- dataset file: `src/dataset_utils/gsm8k.py`
- expected override:
  - `COT_MIMIC_DATASET_SOURCE_GSM8K=/path/to/gsm8k`
- full eval size:
  - `1319`

CommonsenseQA:
- dataset file: `src/dataset_utils/commonsenceqa.py`
- expected override:
  - `COT_MIMIC_DATASET_SOURCE_COMMONSENSEQA=/path/to/commonsenseqa`
- full eval size:
  - `1221`

StrategyQA:
- dataset file: `src/dataset_utils/strategyqa.py`
- expected override:
  - `COT_MIMIC_DATASET_SOURCE_STRATEGYQA=/path/to/strategyqa`
- full eval size:
  - `687`

Suggested setup for a new machine:

```bash
conda env create -f environment.yml
conda activate licv
export COT_MIMIC_RUNNER_PYTHON=/path/to/python
export COT_MIMIC_CONDA_ACTIVATE='source /path/to/conda.sh && conda activate licv'
export COT_MIMIC_MODEL_ROOT_QWEN2_5_7B_INSTRUCT=/path/to/Qwen2.5-7B-Instruct
export COT_MIMIC_DATASET_SOURCE_GSM8K=/path/to/gsm8k
export COT_MIMIC_DATASET_SOURCE_COMMONSENSEQA=/path/to/commonsenseqa
export COT_MIMIC_DATASET_SOURCE_STRATEGYQA=/path/to/strategyqa
```

Important:
- the `/path/to/...` values above are placeholders
- after cloning the repo, users should replace them with the real local paths on their own machine
- if these variables are not set, Codex should help the user set them or pass the corresponding local paths directly

When running the full pipeline, Codex should check and set:
- `model_name`
- model root implied by `src/utils.py` or the matching `COT_MIMIC_MODEL_ROOT_*` override
- `data.name`
- `data.source` when the dataset uses a local source path
- runtime python / conda activation when the local environment differs from the author's machine

Default automation policy:
- do not ask the user to manually edit yaml files
- Codex should update the required yaml parameters itself when the task is clear
- Codex should choose the matching eval yaml automatically:
  - baseline -> `eval_baseline.yaml`
  - base -> `eval_base.yaml`
  - ffn -> `eval_ffn.yaml`
  - attn -> `eval_attn.yaml`
- Codex should then run the stages in order unless the user asks for only one stage
- for a full pipeline request, Codex should include `baseline` even when the requested main vector method is `ffn`, `base`, or `attn`
- the baseline stage is the required no-vector reference and should normally run before generate/extract/vector-eval stages
- every stage must produce a log file so progress is visible
- run the pipeline with the `licv` environment by default
- if a long pipeline fails because of a small dataset/config/runtime bug, Codex may generate a temporary stage-resume script such as `scripts/resume_*.sh` to continue automatically
- this resume script is not a new research stage; it is only an execution recovery tool
- preferred use case:
  - regenerate only the failed stage and then continue to later stages
  - keep logs and outputs in the original run directory
  - reuse the same dataset, model, vector path, and log path unless a collision forces a timestamped filename
- if a resume script is created, Codex should also update this skill or the main controller later so the workaround becomes part of the stable automation path

## Repo files to rely on

Core scripts:
- `src/generate_self_cot.py`
- `src/extract_cot_vector.py`
- `src/eval.py`
- `scripts/run_eval_layers.py`

Static yaml files:
- `src/config/generate_self_cot.yaml`
- `src/config/extract_cot_vector.yaml`
- `src/config/eval_baseline.yaml`
- `src/config/eval_base.yaml`
- `src/config/eval_ffn.yaml`
- `src/config/eval_attn.yaml`

Dataset files for the default examples:
- `src/dataset_utils/gsm8k.py`
- `src/dataset_utils/commonsenceqa.py`
- `src/dataset_utils/strategyqa.py`

Important naming note:
- the repo file is `commonsenceqa.py`
- the config dataset name is `commonsenseqa`

## Three stages

### 1. Generate self-CoT

Use:
- `src/generate_self_cot.py`
- `src/config/generate_self_cot.yaml`

Main fields to modify:
- `model_name`
- `data.name`
- `data.source` if needed
- `data.num_query_samples`
- `output_path`
- `batch_size`
- remove incompatible leftover dataset fields

Expected outputs:
- `.../self_cot_data.json`
- `.../self_cot_data_correct_only.json`

The extraction stage should usually read the `correct_only` file.

### 2. Extract vector

Use:
- `src/extract_cot_vector.py`
- `src/config/extract_cot_vector.yaml`

Main fields to modify:
- `model_name`
- `data.name`
- `data.self_cot_path`
- `output_path`
- `batch_size`
- `encoder`
- remove incompatible leftover dataset fields

Expected output:
- a vector file ending in `.pt`

The extracted file usually contains:
- `ffn_cot_vector`
- `ffn_hs_vector`
- `attn_cot_vector`
- per-layer counts

### 3. Evaluate

Use:
- `src/eval_baseline.py` for baseline
- `src/eval_base.py` for base-vector eval
- `src/eval_licv.py` for FFN / licv eval
- `src/eval_mimic.py` for attn / mimic eval
- `scripts/run_eval_layers_base.py` for base layerwise testing
- `scripts/run_eval_layers_licv.py` for FFN / licv layerwise testing
- `scripts/run_eval_layers_mimic.py` for attn / mimic layerwise testing

Important:
- do not force all methods through the same eval entrypoint
- different methods must call different eval python files
- if you want to test layer by layer, do not hand-roll a loop over `eval_*`
- always use the method-matched layerwise script
- the layerwise script then calls the method-matched eval file
- before layerwise eval, first edit the matching eval yaml
- if a layerwise eval output filename conflicts with an existing file, do not stop
- instead, generate a new filename with a timestamp and continue

Eval yaml roles:
- `eval_baseline.yaml`: no extracted vector
- `eval_base.yaml`: base-vector eval
- `eval_ffn.yaml`: FFN / licv eval
- `eval_attn.yaml`: attn / mimic eval

Entrypoint mapping:
- `baseline` -> `src/eval_baseline.py` + `eval_baseline.yaml`
- `base` -> `src/eval_base.py` + `eval_base.yaml`
- `ffn` -> `src/eval_licv.py` + `eval_ffn.yaml`
- `attn` -> `src/eval_mimic.py` + `eval_attn.yaml`
- layerwise `base` -> `scripts/run_eval_layers_base.py`
- layerwise `ffn` -> `scripts/run_eval_layers_licv.py`
- layerwise `attn` -> `scripts/run_eval_layers_mimic.py`

Startup rules learned from real runs:
- do not collapse all methods into one generic eval file
- do not hand-write a Python loop to replace an existing `run_eval_layers_*` script
- for Hydra overrides, prefer `++key=value` for fields that may or may not already exist
- high-risk fields include:
  - `data.source`
  - `devices`
  - `record_root`
  - `data.query_split`
  - `data.support_split`
- for `record_dir_tag`, use append semantics when the field is not guaranteed to exist yet
  - typical safe form:
    - `+record_dir_tag=...`
- single-modal `qwen2.5-7b-instruct` must use a dedicated Qwen text-model branch in every eval entrypoint
- do not let `qwen2.5-7b-instruct` fall through to a generic `build_model(...)[0].to(...)` path
- every `run_eval_layers_*` script must explicitly pass `model_name` down to the inner `eval_*` script
- otherwise the inner eval may silently fall back to the yaml default model, which can switch a text run into a VL run
- before starting a long run, check the target GPU with `nvidia-smi`
- if the chosen GPU is already occupied, do not assume the code is broken
- first distinguish:
  - real code/config bug
  - GPU OOM or leftover failed process
- if a failed stage leaves a stale GPU process behind, clean that process before retrying
- if a long `tmux` launch command is unstable, Codex may generate a small dedicated launcher script in `scripts/`
- launcher scripts are execution helpers, not research stages
- layerwise eval records must be redirected into the current run directory through `record_root`
- do not allow layerwise json results to fall back to shared `results/record/...`

Main fields to modify:
- `model_name`
- `data.name`
- `extracted_cot_vector_path`
- `use_extracted_cot_vector_type`
- `use_base_vector`
- `start_layer`
- `end_layer`
- `shift_layers_combinations`
- remove incompatible leftover dataset fields

## Default demonstrations

### GSM8K

Use:
- dataset file: `src/dataset_utils/gsm8k.py`
- config value: `data.name=gsm8k`

Typical flow:
1. generate GSM8K self-CoT
2. extract GSM8K vector
3. evaluate GSM8K baseline / base / ffn / attn

Default counts:
- generate: `500`
- eval full set: `1319`

### CommonsenseQA

Use:
- dataset file: `src/dataset_utils/commonsenceqa.py`
- config value: `data.name=commonsenseqa`

Typical flow:
1. generate CommonsenseQA self-CoT
2. extract CommonsenseQA vector
3. evaluate CommonsenseQA baseline / base / ffn / attn

Default counts:
- generate: `500`
- eval full set: `1221`

### StrategyQA

Use:
- dataset file: `src/dataset_utils/strategyqa.py`
- config value: `data.name=strategyqa`

Typical flow:
1. generate StrategyQA self-CoT
2. extract StrategyQA vector
3. evaluate StrategyQA baseline / base / ffn / attn

Default counts:
- generate: `500`
- eval full set: `687`

Important split rule for all supported unimodal datasets:
- the vector-building side should still use the non-test pool with `500` samples by default
- do not accidentally switch generate/extract onto the evaluation test split just because the test set is smaller or easier to count
- eval is the only stage that should use the full held-out test/query set by default
- for `strategyqa` specifically:
  - vector extraction should still come from the non-test side
  - full evaluation should use the held-out test set size `687`

## Parameter mapping

Model switch:
- change `model_name`

Dataset switch:
- change `data.name`
- if needed, also change dataset-specific fields like `source` and `num_query_samples`
- remove or replace leftover fields from the previous dataset

Important:
- do not only change `data.name`
- when switching a yaml from one dataset to another, update the whole `data:` block
- if a yaml still contains fields from the previous dataset, remove or replace them
- for example, if a yaml still has MathVista-style fields like `query_split: testmini` and `support_split: testmini`, those should not be kept when switching to `gsm8k` or `commonsenseqa`

Generate output:
- change `output_path`

Extract input:
- change `data.self_cot_path`

Extract output:
- change `output_path`

Eval vector:
- change `extracted_cot_vector_path`

Eval mode choice:
- baseline: `eval_baseline.yaml`
- base: `eval_base.yaml`
- ffn: `eval_ffn.yaml`
- attn: `eval_attn.yaml`

Layer sweep:
- first edit the matching eval yaml
- then run `scripts/run_eval_layers.py`

Fields that often need to be removed or replaced when changing datasets:
- `query_split`
- `support_split`
- `subsets`
- dataset-specific `source`
- any leftover split or subset field from the previous dataset

## What Codex should change automatically

When the user gives a clear target such as:
- model: `qwen2.5-7b-instruct`
- dataset: `gsm8k`
- method: `baseline`, `base`, `ffn`, or `attn`

Codex should automatically update the relevant yaml fields instead of asking the user to do it by hand.

Generate stage:
- `model_name`
- `data.name`
- dataset-specific `source` if needed
- `data.num_query_samples`, default `500` for this workflow unless the user overrides it
- `output_path`
- `batch_size`

Extract stage:
- `model_name`
- `data.name`
- `data.self_cot_path`
- `output_path`
- `batch_size`
- `encoder`

Eval stage:
- choose the correct eval yaml
- `model_name`
- `data.name`
- `extracted_cot_vector_path`
- `use_extracted_cot_vector_type`
- `use_base_vector`
- `data.num_query_samples`, default to full dataset for this workflow
- `start_layer`
- `end_layer`
- `shift_layers_combinations`

Path naming:
- every run must use a dedicated directory under `results/static_pipeline/`
- the run directory name should include the current task and a timestamp
- generate outputs should go inside that run directory
- extracted vectors should go inside that run directory
- eval records should go inside that run directory
- stage logs should go under that run directory's `logs/` folder
- a good pattern is:
  - `results/static_pipeline/<model>_<dataset>_<timestamp>/`
  - `results/static_pipeline/<model>_<dataset>_<timestamp>/logs/`
  - `results/static_pipeline/<model>_<dataset>_<timestamp>/records/`
- do not send new artifacts to shared global folders if the same run-local path can be used instead

Filename policy:
- if a target output file already exists and the run is supposed to produce a new artifact, do not stop
- generate a new filename with a timestamp and continue
- this also applies to generated self-CoT outputs and vector outputs

Logging policy:
- generate must write a log file
- extract must write a log file
- single eval must write a log file
- layerwise eval should write one combined log file
- inside that log, each layer run should be appended in order
- the log path should be printed before the stage starts

## Layerwise visualization

When the user asks for a summary figure after running `base`, `ffn`, and `attn` layerwise sweeps, Codex should produce a small result package instead of only quoting log lines.

Required outputs:
- one layerwise summary table
- one line plot
- one short best-layer summary

Expected content:
- collect all layers for `base`, `ffn`, and `attn`
- convert accuracy from `[0, 1]` to percentage
- keep exactly 2 decimals
- use `xx.xx` format
- for example:
  - `0.8230958230958231 -> 82.31`

Table requirements:
- rows are `layer0` to `layerN`
- columns are `base`, `ffn`, `attn`
- highlight the best layer for each method if possible
- save the table as an image, not only csv or markdown

Line plot requirements:
- x-axis is layer index
- y-axis is accuracy in percent
- include the three methods:
  - `base`
  - `ffn`
  - `attn`
- include a horizontal baseline line when baseline is available
- annotate or separately report the best result of each method and its layer
- prefer a paper-style figure similar to the existing `picture_code/layerwise.py` aesthetic

Recommended artifact set:
- `layerwise_accuracy_summary.csv`
- `layerwise_accuracy_table.png`
- `layerwise_accuracy_plot.png`
- `layerwise_accuracy_plot.pdf`
- `best_summary.md`

Recommended save location:
- do not put these comparison artifacts under `results/analysis/`
- instead create a new task-level folder that is easy to find later
- the folder name should depend on the dataset
- preferred pattern:
  - `results/static_pipeline/gsm8k_all/`
  - `results/static_pipeline/commonsenseqa_all/`
  - `results/static_pipeline/strategyqa_all/`
- put the comparison table, line plot, csv, and short summary into that dataset-level folder
- if multiple versions are needed, add a timestamped subfolder inside that dataset-level folder instead of mixing with older analysis outputs

Script location policy:
- do not force these comparison scripts into `picture_code/`
- if a dedicated helper script is needed, place it near the current workflow outputs or in a task-specific utility location that does not mix with older shared analysis assets

## Minimal workflow

### Generate

```bash
python src/generate_self_cot.py \
  model_name=qwen2.5-7b-instruct \
  data.name=gsm8k \
  ++data.source=${COT_MIMIC_DATASET_SOURCE_GSM8K} \
  output_path=results/example_runs/gsm8k/self_cot_data.json
```

### Extract

```bash
python src/extract_cot_vector.py \
  --config-name extract_cot_vector \
  model_name=qwen2.5-7b-instruct \
  data.name=gsm8k \
  ++data.source=${COT_MIMIC_DATASET_SOURCE_GSM8K} \
  data.self_cot_path=results/example_runs/gsm8k/self_cot_data_correct_only.json \
  output_path=results/static_pipeline/example_gsm8k_run/example_gsm8k_run.pt
```

### Eval one run

```bash
python src/eval_licv.py \
  --config-name eval_ffn \
  model_name=qwen2.5-7b-instruct \
  data.name=gsm8k \
  ++data.source=${COT_MIMIC_DATASET_SOURCE_GSM8K} \
  use_extracted_cot_vector=True \
  use_extracted_cot_vector_type=licv \
  extracted_cot_vector_path=results/static_pipeline/example_gsm8k_run/example_gsm8k_run.pt \
  eval_mode=EVAL_WITH_COT_VECTOR_DIRECT_Q \
  only_shift_at_layer=15
```

Notes:
- these examples intentionally use repo-relative output paths so they can be copied to another machine without editing author-specific directories
- `${COT_MIMIC_DATASET_SOURCE_*}` stands for the local dataset root on the target machine
- if a user does not have these environment variables set, Codex should either set `++data.source=/path/to/...` directly or use `scripts/run_static_pipeline.py`

### Layerwise eval

```bash
python scripts/run_eval_layers.py
```

Before running that command:
- edit the matching eval yaml first
- for base runs, edit `src/config/eval_base.yaml`
- for ffn runs, edit `src/config/eval_ffn.yaml`
- for attn runs, edit `src/config/eval_attn.yaml`

In normal use, Codex should perform those yaml edits automatically.

## Working style

Prefer this order:
1. read the relevant yaml
2. confirm `model_name`
3. confirm `data.name`
4. confirm input/output paths
5. run the stage
6. verify the artifact before moving to the next stage

For long jobs, prefer `tmux`.

If a long `tmux` command becomes fragile or keeps failing because the command string is too long or too hard to debug:
- Codex may create a small executable recovery script under `scripts/`
- names should be explicit, for example `resume_gsm8k_ffn_pipeline.sh`
- the script should only contain execution glue for the existing stages
- it should not introduce a new algorithmic step
- once the urgent run is stable, Codex should consider merging the logic back into the main controller

## Validation checklist

After generate:
- confirm `self_cot_data_correct_only.json` exists
- confirm the generate log exists

After extract:
- confirm the output is a `.pt` file
- confirm the vector keys exist
- confirm the extract log exists

After eval:
- confirm result json files appear under the current run directory, preferably in `records/`
- confirm the eval log exists

After layerwise eval:
- confirm `run_eval_layers.py` is using the yaml you edited
- confirm records are created for the expected layers
- if filenames would collide, use a timestamped filename and continue
- confirm one layerwise log file is being written and appended

## What is still not fully automatic

The workflow can be mostly automated, but these parts may still need judgment:

- choosing whether the user wants baseline, base, ffn, or attn if they did not say
- choosing sample counts or layer ranges when the user did not specify them
- dataset-specific path fixes if the local dataset layout has changed
- recovering from runtime errors caused by bad checkpoints, missing self-CoT files, or incompatible vector files

Default behavior for ambiguous requests:
- if the user asks for the full pipeline without specifying the eval method, run `baseline` first, then `ffn`
- if the user asks for the full pipeline and does specify a vector method such as `ffn`, `base`, or `attn`, still run `baseline` first unless they explicitly ask to skip it
- if the user asks for layerwise eval without a layer range, use the yaml defaults

## Failure handling

When the pipeline fails, Codex should not stop at the first traceback. It should inspect the error and try the smallest safe fix first.

Preferred fixes:
- if the wrong python environment is used, first try `COT_MIMIC_RUNNER_PYTHON` or `COT_MIMIC_CONDA_ACTIVATE` instead of editing code
- if a deleted optional module such as `llava_ov_model_wrapper` is still imported by unrelated code, remove the hard dependency or make it a lazy import
- if a Hydra override uses the wrong prefix, fix the override and rerun
- if `CUDA_VISIBLE_DEVICES` is set, pass logical GPU indices like `devices=0` to subprocesses rather than the original physical GPU id
- if a required model or dataset path is missing, set the matching `COT_MIMIC_MODEL_ROOT_*` or `COT_MIMIC_DATASET_SOURCE_*` override
- if a filename collides, generate a timestamped new filename and continue

For this default static pipeline:
- Qwen `gsm8k` / `commonsenseqa` runs should not be blocked by deleted LLaVA-specific files
- if the default record filename already exists or collides, create a timestamped filename instead of stopping
