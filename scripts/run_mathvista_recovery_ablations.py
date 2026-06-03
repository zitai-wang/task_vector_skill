#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path("/home/wzy02/work/cot-mimic")
RESULTS_DIR = ROOT / "results"
CKPT_ROOT = RESULTS_DIR / "ckpt"
ABLATION_ROOT = RESULTS_DIR / "recovery_ablations" / "mathvista"
DEFAULT_PYTHON = "python"
DEFAULT_MODEL_NAME = "qwen2.5-vl-7b-instruct"
DEFAULT_DATASET_NAME = "mathvista"
DEFAULT_TRAIN_SAMPLES = 500


def expand_runname(runname: str, num_query_samples: int) -> str:
    return f"{runname}-{DEFAULT_MODEL_NAME}-{DEFAULT_DATASET_NAME}-{num_query_samples}"


def summary_path_for_run(runname: str, num_query_samples: int) -> Path:
    return CKPT_ROOT / expand_runname(runname, num_query_samples) / "recovery_ablation_summary.json"


def run_command(cmd: list[str], env: dict[str, str]) -> None:
    print("COMMAND:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def summarize_init_group(rows: list[dict], threshold: float) -> str:
    if not rows:
        return "初始化消融暂无结果。"

    ordered = sorted(rows, key=lambda x: x["final_accuracy"], reverse=True)
    best = ordered[0]
    row_map = {row["init_mode"]: row for row in rows}
    mm = row_map.get("multimodal_extracted")
    rnd = row_map.get("random")
    uni = row_map.get("unimodal_extracted")

    parts = [
        f"初始化消融中，当前最好的是 {best['init_mode']}，最终 accuracy 为 {best['final_accuracy']:.4f}。"
    ]
    if mm and rnd:
        diff = mm["final_accuracy"] - rnd["final_accuracy"]
        if diff >= threshold:
            parts.append(
                f"multimodal_extracted 相比 random 明显更好，提升 {diff:.4f}。"
            )
        else:
            parts.append(
                f"multimodal_extracted 相比 random 没有明显优势，差值 {diff:.4f}。"
            )
    if uni:
        parts.append(
            f"unimodal_extracted 的最终 accuracy 为 {uni['final_accuracy']:.4f}。"
        )
    return " ".join(parts)


def summarize_loss_group(rows: list[dict], threshold: float) -> str:
    if not rows:
        return "loss 消融暂无结果。"

    ordered = sorted(rows, key=lambda x: x["final_accuracy"], reverse=True)
    best = ordered[0]
    row_map = {row["loss_mode"]: row for row in rows}
    joint = row_map.get("ce_align")
    ce_only = row_map.get("ce_only")
    align_only = row_map.get("align_only")

    parts = [
        f"loss 消融中，当前最好的是 {best['loss_mode']}，最终 accuracy 为 {best['final_accuracy']:.4f}。"
    ]
    if joint and ce_only and align_only:
        best_single = max(ce_only, align_only, key=lambda x: x["final_accuracy"])
        diff = joint["final_accuracy"] - best_single["final_accuracy"]
        if diff >= threshold:
            parts.append(
                f"ce_align 明显优于最强单项损失 {best_single['loss_mode']}，提升 {diff:.4f}。"
            )
        else:
            parts.append(
                f"ce_align 相比最强单项损失 {best_single['loss_mode']} 没有明显优势，差值 {diff:.4f}。"
            )
    return " ".join(parts)


def build_train_cmd(
    python_bin: str,
    runname: str,
    layer: int,
    init_mode: str,
    loss_mode: str,
    num_query_samples: int,
    extra_overrides: list[str],
) -> list[str]:
    return [
        python_bin,
        str(ROOT / "src" / "train.py"),
        "--config-name",
        "train_mathvista_recovery_ablation",
        f"runname={runname}",
        f"only_shift_at_layer={layer}",
        f"shift_layers_combinations=[[{layer}]]",
        f"init_mode={init_mode}",
        f"loss_mode={loss_mode}",
        f"data.num_query_samples={num_query_samples}",
        *extra_overrides,
    ]


def maybe_add_optional_override(overrides: list[str], key: str, value: str | None) -> None:
    if value:
        overrides.append(f"{key}={value}")


def main():
    parser = argparse.ArgumentParser(description="Run MathVista recovery ablations.")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--groups", choices=["init", "loss", "all"], default="all")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--num-query-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--multimodal-path", default=None)
    parser.add_argument("--unimodal-path", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--trend-threshold", type=float, default=0.01)
    args = parser.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'src'}"
    env["TOKENIZERS_PARALLELISM"] = "false"

    extra_overrides: list[str] = []
    maybe_add_optional_override(
        extra_overrides,
        "recovery_vector_init.multimodal_extracted.path",
        args.multimodal_path,
    )
    maybe_add_optional_override(
        extra_overrides,
        "recovery_vector_init.unimodal_extracted.path",
        args.unimodal_path,
    )

    if args.smoke:
        extra_overrides.extend(
            [
                "epochs=1",
                "batch_size=1",
                "accumulate_grad_batches=1",
                "data.num_query_samples=4",
                "data.num_workers=0",
                "validation_eval.max_eval_samples=8",
                "validation_eval.generation_args.max_new_tokens=64",
            ]
        )
        num_query_samples = 4
    else:
        num_query_samples = args.num_query_samples

    init_rows = []
    if args.groups in {"init", "all"}:
        for init_mode in ["random", "multimodal_extracted", "unimodal_extracted"]:
            runname = f"mathvista-recovery-init-{init_mode}-l{args.layer}"
            summary_path = summary_path_for_run(runname, num_query_samples)
            if args.skip_existing and summary_path.exists():
                summary = load_summary(summary_path)
            else:
                cmd = build_train_cmd(
                    python_bin=args.python,
                    runname=runname,
                    layer=args.layer,
                    init_mode=init_mode,
                    loss_mode="ce_align",
                    num_query_samples=num_query_samples,
                    extra_overrides=extra_overrides,
                )
                run_command(cmd, env=env)
                summary = load_summary(summary_path)
            init_rows.append(summary)

        init_payload = {"benchmark": "mathvista", "group": "init", "rows": init_rows}
        init_dir = ABLATION_ROOT / f"layer_{args.layer}"
        write_json(init_dir / "init_summary.json", init_payload)
        write_text(
            init_dir / "init_trend.txt",
            summarize_init_group(init_rows, threshold=args.trend_threshold),
        )

    if args.groups in {"loss", "all"}:
        loss_rows = []
        for loss_mode in ["ce_only", "align_only", "ce_align"]:
            runname = f"mathvista-recovery-loss-{loss_mode}-l{args.layer}"
            summary_path = summary_path_for_run(runname, num_query_samples)
            if args.skip_existing and summary_path.exists():
                summary = load_summary(summary_path)
            else:
                cmd = build_train_cmd(
                    python_bin=args.python,
                    runname=runname,
                    layer=args.layer,
                    init_mode="multimodal_extracted",
                    loss_mode=loss_mode,
                    num_query_samples=num_query_samples,
                    extra_overrides=extra_overrides,
                )
                run_command(cmd, env=env)
                summary = load_summary(summary_path)
            loss_rows.append(summary)

        loss_payload = {"benchmark": "mathvista", "group": "loss", "rows": loss_rows}
        loss_dir = ABLATION_ROOT / f"layer_{args.layer}"
        write_json(loss_dir / "loss_summary.json", loss_payload)
        write_text(
            loss_dir / "loss_trend.txt",
            summarize_loss_group(loss_rows, threshold=args.trend_threshold),
        )


if __name__ == "__main__":
    main()
