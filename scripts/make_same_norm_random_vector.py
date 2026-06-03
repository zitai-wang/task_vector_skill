import argparse
from pathlib import Path

import torch


def same_norm_random_like(tensor: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    base = torch.randn(tensor.shape, generator=generator, dtype=torch.float32)

    if tensor.dim() == 2:
        target_norm = tensor.to(torch.float32).norm(dim=1, keepdim=True)
        rand_norm = base.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return base * (target_norm / rand_norm)

    if tensor.dim() == 3:
        target_norm = tensor.to(torch.float32).norm(dim=2, keepdim=True)
        rand_norm = base.norm(dim=2, keepdim=True).clamp_min(1e-12)
        return base * (target_norm / rand_norm)

    raise ValueError(f"Unsupported tensor rank for same-norm randomization: {tensor.dim()}")


def main():
    parser = argparse.ArgumentParser(description="Create a same-norm random control vector from an extracted vector.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    result = dict(payload)

    for key in ["ffn_cot_vector", "ffn_hs_vector", "attn_cot_vector"]:
        value = payload.get(key)
        if value is None:
            continue
        result[key] = same_norm_random_like(value, seed=args.seed)

    result["control_type"] = "same_norm_random"
    result["control_source_vector"] = str(input_path)
    result["control_seed"] = args.seed

    torch.save(result, output_path)
    print(f"[done] wrote same-norm random vector to {output_path}")


if __name__ == "__main__":
    main()
