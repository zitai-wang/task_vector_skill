#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path("/data1/wzy/cot-mimic")
OUT_DIR = ROOT / "results" / "analysis" / "ffn_hs_vector_pair_sweep_20260428"
KEY = "ffn_hs_vector"

PAIRS = [
    (
        "Intern CommonsenseQA",
        ROOT / "results/average_cot_vectors/internlm2_5-7b-chat_commonsenseqa_correct_only_500_avg_cot_vector_mimic_20260407.pt",
        ROOT / "results/average_cot_vectors/internvl2_5-8b_commonsenseqa_correct_only_500_avg_cot_vector_mimic_20260406.pt",
    ),
    (
        "Intern GSM8K",
        ROOT / "results/average_cot_vectors/internlm2_5-7b-chat_gsm8k_correct_only_500_avg_cot_vector_mimic_20260409_cuda2.pt",
        ROOT / "results/average_cot_vectors/internvl2_5-8b_gsm8k_correct_only_500_avg_cot_vector_mimic_20260408_cuda2.pt",
    ),
    (
        "Intern StrategyQA",
        ROOT / "results/average_cot_vectors/internlm2_5-7b-chat_strategyqa_correct_only_500_avg_cot_vector_mimic_20260406.pt",
        ROOT / "results/average_cot_vectors/internvl2_5-8b_strategyqa_correct_only_500_avg_cot_vector_mimic_20260406.pt",
    ),
    (
        "Qwen CommonsenseQA",
        ROOT / "results/average_cot_vectors/qwen2.5-7b-instruct_commonsenseqa_selfcot401_aligned_avg_cot_vector_mimic_20260417_201129.pt",
        ROOT / "results/average_cot_vectors/qwen2.5-vl-7b-instruct_commonsenseqa_correct_only_500_avg_cot_vector_mimic_20260401.pt",
    ),
    (
        "Qwen GSM8K",
        ROOT / "results/average_cot_vectors/qwen2.5-7b-instruct_gsm8k_selfcot458_aligned_avg_cot_vector_mimic_20260417_161916.pt",
        ROOT / "results/average_cot_vectors/qwen2.5-vl-7b-instruct_gsm8k_correct_only_avg_cot_vector_mimic_20260402_162907.pt",
    ),
    (
        "Qwen StrategyQA",
        ROOT / "results/average_cot_vectors/qwen2.5-7b-instruct_strategyqa_selfcot375_aligned_avg_cot_vector_mimic_20260417_201129.pt",
        ROOT / "results/average_cot_vectors/qwen2.5-vl-7b-instruct_strategyqa_correct_only_500_avg_cot_vector_mimic_20260401.pt",
    ),
]


def load_vector(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensor = payload[KEY].float()
    return tensor.reshape(tensor.shape[0], -1)


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + 1e-12)


def cosine_matrix(x: torch.Tensor) -> np.ndarray:
    y = normalize_rows(x)
    return (y @ y.T).cpu().numpy()


def participation_ratio_mean(x: torch.Tensor) -> float:
    arr = x.cpu().numpy()
    scores = []
    for row in arr:
        sq = row**2
        scores.append(float((sq.sum() ** 2) / ((sq**2).sum() + 1e-12)))
    return float(np.mean(scores))


def cumulative_spectrum(x: torch.Tensor) -> np.ndarray:
    arr = x.cpu().numpy()
    arr = arr - arr.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(arr, compute_uv=False)
    energy = sv**2
    energy = energy / energy.sum()
    return np.cumsum(energy)


def summarize_pair(label: str, llm_path: Path, vlm_path: Path) -> dict[str, float | str]:
    llm = load_vector(llm_path)
    vlm = load_vector(vlm_path)

    llm_cm = cosine_matrix(llm)
    vlm_cm = cosine_matrix(vlm)

    llm_adj = float(np.mean([llm_cm[i, i + 1] for i in range(llm_cm.shape[0] - 1)]))
    vlm_adj = float(np.mean([vlm_cm[i, i + 1] for i in range(vlm_cm.shape[0] - 1)]))
    llm_off = float(llm_cm[np.triu_indices(llm_cm.shape[0], 1)].mean())
    vlm_off = float(vlm_cm[np.triu_indices(vlm_cm.shape[0], 1)].mean())
    llm_pr = participation_ratio_mean(llm)
    vlm_pr = participation_ratio_mean(vlm)

    llm_spec = cumulative_spectrum(llm)
    vlm_spec = cumulative_spectrum(vlm)

    same = (normalize_rows(llm) * normalize_rows(vlm)).sum(dim=1).cpu().numpy()
    early = float(same[:10].mean())
    late = float(same[-10:].mean())
    drop = early - late

    # Positive score means "LLM more structured / less collapsed" than VLM.
    score = (
        (vlm_adj - llm_adj)
        + (vlm_off - llm_off)
        + (vlm_spec[0] - llm_spec[0])
        + (vlm_spec[2] - llm_spec[2])
        + ((llm_pr - vlm_pr) / 1000.0)
        + drop
    )

    return {
        "pair": label,
        "layers": int(llm.shape[0]),
        "adj_delta": vlm_adj - llm_adj,
        "offdiag_delta": vlm_off - llm_off,
        "pr_delta": llm_pr - vlm_pr,
        "sv1_delta": float(vlm_spec[0] - llm_spec[0]),
        "sv3_delta": float(vlm_spec[2] - llm_spec[2]),
        "same_early": early,
        "same_late": late,
        "same_drop": drop,
        "score": float(score),
    }


def plot(results: list[dict[str, float | str]]) -> Path:
    labels = [r["pair"] for r in results]
    score = np.array([float(r["score"]) for r in results])
    adj = np.array([float(r["adj_delta"]) for r in results])
    off = np.array([float(r["offdiag_delta"]) for r in results])
    pr = np.array([float(r["pr_delta"]) / 300.0 for r in results])
    spec = np.array([(float(r["sv1_delta"]) + float(r["sv3_delta"])) / 2.0 for r in results])

    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), gridspec_kw={"width_ratios": [1.0, 1.2]})

    ax = axes[0]
    colors = ["#3366cc" if s > 0 else "#c54e52" for s in score]
    ax.barh(y, score, color=colors)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Overall micro-gap score")
    ax.set_title("(a) Ranking of LLM > VLM micro evidence", fontweight="bold")
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    width = 0.18
    x = np.arange(len(labels))
    ax.bar(x - 1.5 * width, adj, width=width, color="#4c78a8", label="Adj cosine delta")
    ax.bar(x - 0.5 * width, off, width=width, color="#72b7b2", label="Offdiag cosine delta")
    ax.bar(x + 0.5 * width, pr, width=width, color="#54a24b", label="PR delta / 300")
    ax.bar(x + 1.5 * width, spec, width=width, color="#f58518", label="Mean spectral delta")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_title("(b) Component-wise micro signals", fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    out_path = OUT_DIR / "ffn_hs_vector_pair_sweep.png"
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [summarize_pair(*pair) for pair in PAIRS]
    results = sorted(results, key=lambda x: float(x["score"]), reverse=True)

    tsv_path = OUT_DIR / "ffn_hs_vector_pair_sweep.tsv"
    header = [
        "pair",
        "layers",
        "adj_delta",
        "offdiag_delta",
        "pr_delta",
        "sv1_delta",
        "sv3_delta",
        "same_early",
        "same_late",
        "same_drop",
        "score",
    ]
    lines = ["\t".join(header)]
    for row in results:
        lines.append("\t".join(str(row[k]) for k in header))
    tsv_path.write_text("\n".join(lines), encoding="utf-8")

    fig_path = plot(results)
    print(tsv_path)
    print(fig_path)


if __name__ == "__main__":
    main()
