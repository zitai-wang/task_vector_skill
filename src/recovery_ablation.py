import os
from typing import Dict, List, Optional, Tuple

import torch


VALID_INIT_MODES = {"random", "multimodal_extracted", "unimodal_extracted"}
VALID_LOSS_MODES = {"ce_only", "align_only", "ce_align"}


def _as_plain_list(value) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    try:
        return [int(x) for x in value]
    except TypeError:
        return None


def tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    flat = tensor.detach().float().reshape(-1)
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        "norm": float(flat.norm().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


def get_trainable_vector_parameter(encoder) -> Tuple[str, torch.nn.Parameter]:
    if hasattr(encoder, "attn_shift"):
        return "attn_shift", encoder.attn_shift
    if hasattr(encoder, "ffn_shift"):
        return "ffn_shift", encoder.ffn_shift
    raise ValueError("No trainable recovery vector parameter found on encoder.")


def get_trainable_vector_layer_indices(encoder, parameter: torch.nn.Parameter) -> List[int]:
    shift_layers = _as_plain_list(getattr(encoder, "shift_layers", None))
    if shift_layers is not None:
        return shift_layers

    if parameter.dim() >= 1 and parameter.shape[0] == int(getattr(encoder, "lmm_layers")):
        return list(range(int(getattr(encoder, "lmm_layers"))))

    only_shift_at_layer = _as_plain_list(getattr(encoder, "only_shift_at_layer", None))
    if only_shift_at_layer is not None and len(only_shift_at_layer) == parameter.shape[0]:
        return only_shift_at_layer

    raise ValueError(
        "Unable to infer layer indices for the recovery vector. "
        f"parameter_shape={tuple(parameter.shape)}"
    )


def get_trainable_vector_stats(encoder) -> Dict[str, object]:
    parameter_name, parameter = get_trainable_vector_parameter(encoder)
    vector = parameter.detach().float()
    flat = vector.reshape(vector.shape[0], -1)
    per_layer_norms = [float(x.item()) for x in flat.norm(dim=1)]
    return {
        "parameter_name": parameter_name,
        "shape": list(vector.shape),
        "mean": float(vector.mean().item()),
        "std": float(vector.std(unbiased=False).item()) if vector.numel() > 1 else 0.0,
        "norm": float(vector.reshape(-1).norm().item()),
        "per_layer_norms": per_layer_norms,
    }


def _vector_key_for_parameter(parameter_name: str, recovery_vector_semantics: str) -> str:
    if parameter_name == "attn_shift":
        return "attn_cot_vector"
    if parameter_name == "ffn_shift":
        if recovery_vector_semantics == "base":
            return "ffn_hs_vector"
        return "ffn_cot_vector"
    raise ValueError(f"Unsupported recovery parameter: {parameter_name}")


def _slice_loaded_vector(
    loaded_vector: torch.Tensor,
    target_shape: Tuple[int, ...],
    layer_indices: List[int],
) -> torch.Tensor:
    if tuple(loaded_vector.shape) == tuple(target_shape):
        return loaded_vector

    if loaded_vector.dim() == len(target_shape) and loaded_vector.shape[1:] == target_shape[1:]:
        max_layer_idx = max(layer_indices)
        if loaded_vector.shape[0] > max_layer_idx:
            index = torch.tensor(layer_indices, dtype=torch.long, device=loaded_vector.device)
            sliced = loaded_vector.index_select(0, index)
            if tuple(sliced.shape) == tuple(target_shape):
                return sliced

    if loaded_vector.dim() + 1 == len(target_shape) and len(layer_indices) == 1:
        candidate = loaded_vector.unsqueeze(0)
        if tuple(candidate.shape) == tuple(target_shape):
            return candidate

    raise ValueError(
        "Shape mismatch while slicing extracted recovery vector: "
        f"loaded_shape={tuple(loaded_vector.shape)}, target_shape={tuple(target_shape)}, "
        f"layer_indices={layer_indices}"
    )


def resolve_recovery_loss_weights(cfg) -> Tuple[str, float, float]:
    base_ce = float(getattr(cfg, "ce_loss_weight", 1.0))
    base_align = float(getattr(cfg, "align_loss_weight", 1.0))
    loss_mode = getattr(cfg, "loss_mode", None)

    if not loss_mode:
        loss_mode = "align_only" if bool(getattr(cfg, "only_alignment_loss", False)) else "ce_align"

    if loss_mode not in VALID_LOSS_MODES:
        raise ValueError(
            f"Unsupported loss_mode={loss_mode!r}. Supported values: {sorted(VALID_LOSS_MODES)}"
        )

    if loss_mode == "ce_only":
        return loss_mode, base_ce, 0.0
    if loss_mode == "align_only":
        return loss_mode, 0.0, base_align
    return loss_mode, base_ce, base_align


def _get_init_cfg(cfg, init_mode: str):
    init_cfg = getattr(cfg, "recovery_vector_init", None)
    if init_cfg is None:
        return None
    return getattr(init_cfg, init_mode, None)


def apply_recovery_vector_initialization(cfg, encoder) -> Dict[str, object]:
    init_mode = str(getattr(cfg, "init_mode", "random"))
    if init_mode not in VALID_INIT_MODES:
        raise ValueError(
            f"Unsupported init_mode={init_mode!r}. Supported values: {sorted(VALID_INIT_MODES)}"
        )

    parameter_name, parameter = get_trainable_vector_parameter(encoder)
    recovery_vector_semantics = str(getattr(cfg, "recovery_vector_semantics", "mimic"))
    if recovery_vector_semantics == "mimic" and parameter_name != "attn_shift":
        raise ValueError(
            "Mimic recovery semantics require AttnApproximator/attn_shift initialization, "
            f"but got parameter_name={parameter_name!r}."
        )
    layer_indices = get_trainable_vector_layer_indices(encoder, parameter)
    info = {
        "init_mode": init_mode,
        "recovery_vector_semantics": recovery_vector_semantics,
        "parameter_name": parameter_name,
        "target_shape": list(parameter.shape),
        "layer_indices": layer_indices,
        "loaded": False,
        "source_path": None,
        "source_model_name": None,
        "source_dataset_name": None,
    }

    with torch.no_grad():
        if init_mode == "random":
            random_std = float(getattr(cfg, "random_init_std", 1e-3))
            parameter.data.normal_(mean=0.0, std=random_std)
            info["loaded"] = True
            info["source_model_name"] = "random_small_value_init"
            info["source_dataset_name"] = None
        else:
            mode_cfg = _get_init_cfg(cfg, init_mode)
            if mode_cfg is None:
                raise ValueError(f"Missing recovery_vector_init.{init_mode} configuration.")

            source_path = str(getattr(mode_cfg, "path", "") or "").strip()
            if not source_path:
                raise ValueError(f"Missing extracted vector path for init_mode={init_mode}.")
            if not os.path.exists(source_path):
                raise FileNotFoundError(
                    f"Extracted vector file not found for init_mode={init_mode}: {source_path}"
                )

            payload = torch.load(source_path, map_location="cpu", weights_only=False)
            vector_key = _vector_key_for_parameter(parameter_name, recovery_vector_semantics)
            if vector_key not in payload or payload[vector_key] is None:
                raise KeyError(
                    f"Vector key {vector_key!r} not found in extracted vector payload: {source_path}"
                )

            loaded_vector = payload[vector_key].detach().float()
            sliced_vector = _slice_loaded_vector(
                loaded_vector=loaded_vector,
                target_shape=tuple(parameter.shape),
                layer_indices=layer_indices,
            )
            parameter.data.copy_(sliced_vector.to(parameter.device, dtype=parameter.dtype))

            info["loaded"] = True
            info["source_path"] = source_path
            info["source_model_name"] = getattr(mode_cfg, "source_model_name", None)
            info["source_dataset_name"] = getattr(mode_cfg, "source_dataset_name", None)
            info["source_vector_key"] = vector_key

    stats = tensor_stats(parameter.data)
    info.update(
        {
            "shape": list(parameter.shape),
            "mean": stats["mean"],
            "std": stats["std"],
            "norm": stats["norm"],
        }
    )

    print(
        "[RecoveryInit] "
        f"mode={info['init_mode']} "
        f"parameter={info['parameter_name']} "
        f"vector_key={info.get('source_vector_key')} "
        f"shape={tuple(info['shape'])} "
        f"mean={info['mean']:.6f} "
        f"norm={info['norm']:.6f} "
        f"loaded={info['loaded']} "
        f"source={info['source_path']}"
    )
    return info
