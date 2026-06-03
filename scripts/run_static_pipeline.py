#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_RUNNER_PYTHON = "/home/wzy/anaconda3/envs/licv/bin/python"

MODEL_PATHS = {
    "qwen2.5-7b-instruct": "/data/share/model_weight/qwen/Qwen2.5-7B-Instruct/",
}

DATASET_SOURCES = {
    "gsm8k": "/data/share/datasets/gsm8k",
    "commonsenseqa": "/data/share/commonsenceqa/",
    "strategyqa": "/data/share/strategy_qa/",
}

FULL_EVAL_SAMPLES = {
    "gsm8k": 1319,
    "commonsenseqa": 1221,
    "strategyqa": 687,
}

EVAL_CONFIGS = {
    "baseline": "eval_baseline",
    "base": "eval_base",
    "ffn": "eval_ffn",
    "attn": "eval_attn",
}

EVAL_ENTRYPOINTS = {
    "baseline": SRC_DIR / "eval_baseline.py",
    "base": SRC_DIR / "eval_base.py",
    "ffn": SRC_DIR / "eval_licv.py",
    "attn": SRC_DIR / "eval_mimic.py",
}

LAYERWISE_ENTRYPOINTS = {
    "base": PROJECT_ROOT / "scripts" / "run_eval_layers_base.py",
    "ffn": PROJECT_ROOT / "scripts" / "run_eval_layers_licv.py",
    "attn": PROJECT_ROOT / "scripts" / "run_eval_layers_mimic.py",
}


@dataclass
class PipelinePaths:
    run_tag: str
    output_dir: Path
    logs_dir: Path
    records_dir: Path
    self_cot_all: Path
    self_cot_correct: Path
    vector_pt: Path
    summary_json: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the cot-mimic static pipeline: generate -> extract -> eval."
    )
    parser.add_argument(
        "--model-name",
        default="qwen2.5-7b-instruct",
        choices=sorted(MODEL_PATHS.keys()),
    )
    parser.add_argument(
        "--dataset",
        default="gsm8k",
        choices=sorted(DATASET_SOURCES.keys()),
    )
    parser.add_argument(
        "--method",
        default="ffn",
        choices=["baseline", "base", "ffn", "attn"],
        help="Eval method. baseline skips vector use; others use the extracted vector.",
    )
    parser.add_argument("--devices", default=None, help="Visible GPU ids, e.g. 0 or 0,1")
    parser.add_argument("--generate-samples", type=int, default=500)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--generate-batch-size", type=int, default=8)
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=28)
    parser.add_argument(
        "--no-layerwise",
        action="store_true",
        help="Run a single eval instead of sweeping layers.",
    )
    parser.add_argument(
        "--single-layer",
        type=int,
        default=None,
        help="Used only with --no-layerwise. Defaults to start-layer.",
    )
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional custom run tag. Defaults to a timestamped tag.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_unique_run_tag(base_tag: str) -> str:
    candidate = base_tag
    output_dir = PROJECT_ROOT / "results" / "static_pipeline" / candidate
    if not output_dir.exists():
        return candidate
    return f"{base_tag}_{timestamp()}"


def build_paths(args: argparse.Namespace) -> PipelinePaths:
    base_tag = args.run_tag or f"{args.model_name}_{args.dataset}_{timestamp()}"
    run_tag = ensure_unique_run_tag(base_tag)
    output_dir = PROJECT_ROOT / "results" / "static_pipeline" / run_tag
    logs_dir = output_dir / "logs"
    records_dir = output_dir / "records"
    self_cot_all = output_dir / "self_cot_data.json"
    self_cot_correct = output_dir / "self_cot_data_correct_only.json"
    vector_pt = output_dir / f"{run_tag}.pt"
    summary_json = output_dir / "run_summary.json"

    ensure_parent(self_cot_all)
    logs_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(vector_pt)
    return PipelinePaths(
        run_tag=run_tag,
        output_dir=output_dir,
        logs_dir=logs_dir,
        records_dir=records_dir,
        self_cot_all=self_cot_all,
        self_cot_correct=self_cot_correct,
        vector_pt=vector_pt,
        summary_json=summary_json,
    )


def run_command(
    command: List[str],
    env: dict,
    dry_run: bool,
    log_path: Path,
    append: bool = False,
    stage_label: str | None = None,
) -> None:
    print("\n[run_static_pipeline] Running:")
    print(" ".join(command))
    print(f"[run_static_pipeline] log={log_path}")
    if dry_run:
        return
    ensure_parent(log_path)
    mode = "a" if append else "w"
    with open(log_path, mode, encoding="utf-8") as log_handle:
        if stage_label:
            log_handle.write(f"\n===== {stage_label} =====\n")
        log_handle.write("COMMAND:\n")
        log_handle.write(" ".join(command) + "\n\n")
        log_handle.flush()
        subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            check=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )


def build_env(args: argparse.Namespace) -> dict:
    env = os.environ.copy()
    if args.devices:
        env["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    return env


def runner_python() -> str:
    if os.path.exists(DEFAULT_RUNNER_PYTHON):
        return DEFAULT_RUNNER_PYTHON
    return sys.executable


def subprocess_devices_value(args: argparse.Namespace) -> str | None:
    if args.devices is None:
        return None
    visible = [part.strip() for part in str(args.devices).split(",") if part.strip()]
    if not visible:
        return None
    # After CUDA_VISIBLE_DEVICES is set, subprocesses should use logical indices,
    # not the original physical GPU ids. For our pipeline we bind to the first
    # visible GPU by default.
    return "0"


def add_common_overrides(
    args: argparse.Namespace,
    sample_count: int,
) -> List[str]:
    overrides = [
        f"model_name={args.model_name}",
        f"data.name={args.dataset}",
        f"data.num_query_samples={sample_count}",
    ]
    overrides.append(f"++data.source={DATASET_SOURCES[args.dataset]}")
    logical_devices = subprocess_devices_value(args)
    if logical_devices is not None:
        overrides.append(f"++devices={logical_devices}")
    return overrides


def run_generate(args: argparse.Namespace, paths: PipelinePaths, env: dict) -> None:
    command = [
        runner_python(),
        str(SRC_DIR / "generate_self_cot.py"),
        "--config-name",
        "generate_self_cot",
        *add_common_overrides(args, args.generate_samples),
        f"output_path={paths.self_cot_all}",
        f"max_samples={args.generate_samples}",
        f"batch_size={args.generate_batch_size}",
    ]
    run_command(command, env, args.dry_run, paths.logs_dir / "01_generate.log")


def run_extract(args: argparse.Namespace, paths: PipelinePaths, env: dict) -> None:
    command = [
        runner_python(),
        str(SRC_DIR / "extract_cot_vector.py"),
        "--config-name",
        "extract_cot_vector",
        *add_common_overrides(args, args.generate_samples),
        "data.use_self_cot=True",
        f"data.self_cot_path={paths.self_cot_correct}",
        f"output_path={paths.vector_pt}",
        f"batch_size={args.extract_batch_size}",
    ]
    run_command(command, env, args.dry_run, paths.logs_dir / "02_extract.log")


def eval_config_name(method: str) -> str:
    return EVAL_CONFIGS[method]


def eval_entrypoint(method: str) -> Path:
    return EVAL_ENTRYPOINTS[method]


def layerwise_entrypoint(method: str) -> Path:
    return LAYERWISE_ENTRYPOINTS[method]


def run_single_eval(args: argparse.Namespace, paths: PipelinePaths, env: dict) -> None:
    single_layer = args.single_layer if args.single_layer is not None else args.start_layer
    sample_count = args.eval_samples or FULL_EVAL_SAMPLES[args.dataset]
    command = [
        runner_python(),
        str(eval_entrypoint(args.method)),
        "--config-name",
        eval_config_name(args.method),
        *add_common_overrides(args, sample_count),
        f"batch_size={args.eval_batch_size}",
        f"++record_root={paths.records_dir}",
        f"only_shift_at_layer={single_layer}",
        "resume=False",
    ]
    if args.method != "baseline":
        command.extend(
            [
                "use_extracted_cot_vector=True",
                f"extracted_cot_vector_path={paths.vector_pt}",
            ]
        )
    run_command(command, env, args.dry_run, paths.logs_dir / "03_eval_single.log")


def run_baseline_eval(args: argparse.Namespace, paths: PipelinePaths, env: dict) -> None:
    sample_count = args.eval_samples or FULL_EVAL_SAMPLES[args.dataset]
    command = [
        runner_python(),
        str(eval_entrypoint("baseline")),
        "--config-name",
        eval_config_name("baseline"),
        *add_common_overrides(args, sample_count),
        f"batch_size={args.eval_batch_size}",
        f"++record_root={paths.records_dir}",
        "resume=False",
    ]
    run_command(command, env, args.dry_run, paths.logs_dir / "00_baseline.log")


def run_layerwise_eval(args: argparse.Namespace, paths: PipelinePaths, env: dict) -> None:
    if args.method == "baseline":
        raise ValueError("Layerwise sweep is not supported for baseline method.")

    sample_count = args.eval_samples or FULL_EVAL_SAMPLES[args.dataset]
    command = [
        runner_python(),
        str(layerwise_entrypoint(args.method)),
        *add_common_overrides(args, sample_count),
        f"batch_size={args.eval_batch_size}",
        f"resume=False",
        f"start_layer={args.start_layer}",
        f"end_layer={args.end_layer}",
        f"+record_dir_tag={paths.run_tag}",
        f"++record_root={paths.records_dir}",
        "use_extracted_cot_vector=True",
        f"extracted_cot_vector_path={paths.vector_pt}",
    ]
    run_command(command, env, args.dry_run, paths.logs_dir / "03_eval_layerwise.log")


def verify_outputs(args: argparse.Namespace, paths: PipelinePaths) -> None:
    if args.dry_run:
        return
    if not args.skip_generate and not paths.self_cot_correct.exists():
        raise FileNotFoundError(f"Missing correct-only self-CoT file: {paths.self_cot_correct}")
    if not args.skip_extract and not paths.vector_pt.exists():
        raise FileNotFoundError(f"Missing extracted vector file: {paths.vector_pt}")


def write_summary(args: argparse.Namespace, paths: PipelinePaths) -> None:
    ensure_parent(paths.summary_json)
    summary = {
        "run_tag": paths.run_tag,
        "model_name": args.model_name,
        "dataset": args.dataset,
        "method": args.method,
        "devices": args.devices,
        "subprocess_devices": subprocess_devices_value(args),
        "generate_samples": args.generate_samples,
        "eval_samples": args.eval_samples or FULL_EVAL_SAMPLES[args.dataset],
        "generate_batch_size": args.generate_batch_size,
        "extract_batch_size": args.extract_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "start_layer": args.start_layer,
        "end_layer": args.end_layer,
        "layerwise": not args.no_layerwise,
        "model_path": MODEL_PATHS[args.model_name],
        "data_source": DATASET_SOURCES[args.dataset],
        "self_cot_all": str(paths.self_cot_all),
        "self_cot_correct": str(paths.self_cot_correct),
        "vector_pt": str(paths.vector_pt),
        "records_dir": str(paths.records_dir),
        "logs_dir": str(paths.logs_dir),
        "runner_python": runner_python(),
    }
    with open(paths.summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if args.method == "baseline":
        args.skip_extract = True
    paths = build_paths(args)
    env = build_env(args)

    print(f"[run_static_pipeline] run_tag={paths.run_tag}")
    print(f"[run_static_pipeline] model={args.model_name}")
    print(f"[run_static_pipeline] dataset={args.dataset}")
    print(f"[run_static_pipeline] method={args.method}")

    if not args.skip_eval and not args.skip_baseline and args.method != "baseline":
        run_baseline_eval(args, paths, env)
    if not args.skip_generate:
        run_generate(args, paths, env)
    if not args.skip_extract:
        run_extract(args, paths, env)
    if not args.skip_eval:
        if args.no_layerwise:
            run_single_eval(args, paths, env)
        else:
            run_layerwise_eval(args, paths, env)

    verify_outputs(args, paths)
    write_summary(args, paths)
    print(f"\n[run_static_pipeline] Summary written to {paths.summary_json}")


if __name__ == "__main__":
    main()
