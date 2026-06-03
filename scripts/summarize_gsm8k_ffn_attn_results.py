#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/data1/wzy/cot-mimic")
FFN_RECORD_DIR = ROOT / "results/static_pipeline/qwen2.5-7b-instruct_gsm8k_ffn_sweep_20260527_113950/records"
ATTN_RECORD_DIR = ROOT / "results/static_pipeline/qwen2.5-7b-instruct_gsm8k_attn_20260527_170850_20260527_183104/records"
BASELINE_PATH = ATTN_RECORD_DIR / "0shot.json"
OUT_ROOT = ROOT / "results/static_pipeline/gsm8k_all"


def load_baseline_percent() -> float:
    data = json.loads(BASELINE_PATH.read_text())
    return float(data["eval_result"]["accuracy"]) * 100.0


def load_method_scores(record_dir: Path, prefix: str) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    pattern = re.compile(rf"{re.escape(prefix)}_layers_(\d+)_direct_q_1\.0\.json$")
    for path in sorted(record_dir.glob(f"{prefix}_layers_*_direct_q_1.0.json")):
        match = pattern.search(path.name)
        if not match:
            continue
        layer = int(match.group(1))
        data = json.loads(path.read_text())
        acc = float(data["eval_result"]["accuracy"]) * 100.0
        rows.append((layer, acc))
    return sorted(rows, key=lambda x: x[0])


def format_pct(value: float) -> str:
    return f"{value:.2f}"


def build_table_rows(
    baseline: float,
    ffn_rows: list[tuple[int, float]],
    attn_rows: list[tuple[int, float]],
) -> list[dict[str, float]]:
    attn_map = dict(attn_rows)
    table = []
    for layer, ffn_acc in ffn_rows:
        table.append(
            {
                "layer": layer,
                "baseline": baseline,
                "ffn": ffn_acc,
                "attn": attn_map[layer],
            }
        )
    return table


def save_csv(out_dir: Path, rows: list[dict[str, float]]) -> Path:
    csv_path = out_dir / "layerwise_accuracy_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "baseline", "ffn", "attn"])
        for row in rows:
            writer.writerow(
                [
                    f"layer{row['layer']}",
                    format_pct(row["baseline"]),
                    format_pct(row["ffn"]),
                    format_pct(row["attn"]),
                ]
            )
    return csv_path


def save_table_png(out_dir: Path, rows: list[dict[str, float]]) -> Path:
    fig_h = max(8.0, 0.34 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(7.5, fig_h))
    ax.axis("off")

    best_ffn_idx = max(range(len(rows)), key=lambda i: rows[i]["ffn"])
    best_attn_idx = max(range(len(rows)), key=lambda i: rows[i]["attn"])

    col_labels = ["Layer", "Baseline", "FFN", "Attn"]
    cell_text = [
        [
            f"layer{row['layer']}",
            format_pct(row["baseline"]),
            format_pct(row["ffn"]),
            format_pct(row["attn"]),
        ]
        for row in rows
    ]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2f5c85")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.set_height(cell.get_height() * 1.15)
        else:
            if r - 1 == best_ffn_idx and c == 2:
                cell.set_facecolor("#f7c8c8")
            elif r - 1 == best_attn_idx and c == 3:
                cell.set_facecolor("#cdeccf")
            elif c == 1:
                cell.set_facecolor("#f1f1f1")

    fig.tight_layout()
    path = out_dir / "layerwise_accuracy_table.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def save_plot(out_dir: Path, baseline: float, ffn_rows: list[tuple[int, float]], attn_rows: list[tuple[int, float]]) -> tuple[Path, Path]:
    layers = np.array([x for x, _ in ffn_rows], dtype=np.int32)
    ffn = np.array([y for _, y in ffn_rows], dtype=np.float32)
    attn = np.array([y for _, y in attn_rows], dtype=np.float32)

    best_ffn_layer = int(layers[np.argmax(ffn)])
    best_attn_layer = int(layers[np.argmax(attn)])
    best_ffn = float(np.max(ffn))
    best_attn = float(np.max(attn))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.axhline(baseline, color="#7a7a7a", linestyle="--", linewidth=2.4, label=f"Baseline ({baseline:.2f})")
    ax.plot(layers, ffn, color="#d62728", marker="o", linewidth=3.0, markersize=6.5, label="FFN")
    ax.plot(layers, attn, color="#2ca02c", marker="s", linewidth=3.0, markersize=6.0, label="Attn")

    ax.scatter([best_ffn_layer], [best_ffn], color="#d62728", s=80, zorder=5)
    ax.scatter([best_attn_layer], [best_attn], color="#2ca02c", s=80, zorder=5)
    ax.annotate(f"FFN best L{best_ffn_layer}: {best_ffn:.2f}", (best_ffn_layer, best_ffn),
                xytext=(8, 10), textcoords="offset points", color="#d62728", fontsize=10)
    ax.annotate(f"Attn best L{best_attn_layer}: {best_attn:.2f}", (best_attn_layer, best_attn),
                xytext=(8, -18), textcoords="offset points", color="#2ca02c", fontsize=10)

    ax.set_title("GSM8K Layerwise Accuracy Summary")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(np.arange(0, 28, 2))
    ymin = min(baseline, ffn.min(), attn.min()) - 0.4
    ymax = max(baseline, ffn.max(), attn.max()) + 0.4
    ax.set_ylim(ymin, ymax)
    ax.grid(True, color="#c7c7c7", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    png_path = out_dir / "layerwise_accuracy_plot.png"
    pdf_path = out_dir / "layerwise_accuracy_plot.pdf"
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def save_summary_md(out_dir: Path, baseline: float, ffn_rows: list[tuple[int, float]], attn_rows: list[tuple[int, float]]) -> Path:
    best_ffn_layer, best_ffn = max(ffn_rows, key=lambda x: x[1])
    best_attn_layer, best_attn = max(attn_rows, key=lambda x: x[1])
    delta_ffn = best_ffn - baseline
    delta_attn = best_attn - baseline

    lines = [
        "# GSM8K FFN/Attn Layerwise Summary",
        "",
        f"- Baseline: `{baseline:.2f}%`",
        f"- Best FFN: `layer{best_ffn_layer}` -> `{best_ffn:.2f}%` (`{delta_ffn:+.2f}` vs baseline)",
        f"- Best Attn: `layer{best_attn_layer}` -> `{best_attn:.2f}%` (`{delta_attn:+.2f}` vs baseline)",
        "",
        "## Notes",
        "",
        "- This package summarizes the completed `ffn` and `attn` layerwise sweeps on GSM8K.",
        "- `base` is not included here because it was not part of this finished result set.",
        f"- FFN best layer is `layer{best_ffn_layer}`.",
        f"- Attn best layer is `layer{best_attn_layer}`.",
        "- Attn slightly outperforms both baseline and FFN in this run.",
    ]
    path = out_dir / "best_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline_percent()
    ffn_rows = load_method_scores(FFN_RECORD_DIR, "licv")
    attn_rows = load_method_scores(ATTN_RECORD_DIR, "mimic")
    rows = build_table_rows(baseline, ffn_rows, attn_rows)

    csv_path = save_csv(out_dir, rows)
    table_path = save_table_png(out_dir, rows)
    png_path, pdf_path = save_plot(out_dir, baseline, ffn_rows, attn_rows)
    summary_path = save_summary_md(out_dir, baseline, ffn_rows, attn_rows)

    print(csv_path)
    print(table_path)
    print(png_path)
    print(pdf_path)
    print(summary_path)


if __name__ == "__main__":
    main()
