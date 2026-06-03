import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
os.environ["MAX_PIXELS"] = str(512 * 28 * 28)  # 限制在约 40w 像素

import random
import re
import torch
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

import paths  # noqa: F401
import gc

from src.utils import build_model
from qwen_model_wrapper import QwenModelWrapper
from qwen_vl_model_wrapper import QwenVLModelWrapper
from dataset_utils import dataset_mapping
from shift_encoder import AttnApproximator, ShiftStrategy


def remove_hook_dict(hook_dict):
    if hook_dict is None:
        return
    for _, h in hook_dict.items():
        if isinstance(h, list):
            for x in h:
                if x is not None:
                    x.remove()
        else:
            if h is not None:
                h.remove()


def build_student_answer_mask(batch, device):
    if "student_answer_mask" in batch:
        return batch["student_answer_mask"].to(device).bool()

    if "student_labels" in batch:
        return batch["student_labels"].to(device).ne(-100)

    raise KeyError(
        "Missing `student_answer_mask` in batch. "
        "Extraction requires dataset to return a strict answer-region mask."
    )


def build_teacher_answer_mask(batch, device, tokenizer=None):
    if "teacher_answer_mask" in batch:
        teacher_mask = batch["teacher_answer_mask"].to(device).bool()
        if teacher_mask.dim() == 2 and "prefix_inputs" in batch and "student_labels" in batch:
            teacher_input_ids = batch["prefix_inputs"]["input_ids"].to(device)
            student_labels = batch["student_labels"].to(device)
            pad_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
            eos_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None

            for sample_idx in range(teacher_mask.shape[0]):
                if teacher_mask[sample_idx].any():
                    continue

                answer_ids = student_labels[sample_idx][student_labels[sample_idx].ne(-100)]
                if answer_ids.numel() == 0:
                    continue

                if pad_id is not None:
                    answer_ids = answer_ids[answer_ids.ne(pad_id)]
                if eos_id is not None:
                    answer_ids = answer_ids[answer_ids.ne(eos_id)]
                if answer_ids.numel() == 0:
                    continue

                search_candidates = [answer_ids]

                if tokenizer is not None:
                    answer_text = tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip()
                    if answer_text:
                        text_variants = [answer_text, " " + answer_text, "\n" + answer_text, "#### " + answer_text]
                        for text_variant in text_variants:
                            variant_ids = tokenizer(text_variant, add_special_tokens=False).input_ids
                            if variant_ids:
                                search_candidates.append(
                                    torch.tensor(variant_ids, device=device, dtype=teacher_input_ids.dtype)
                                )

                full_ids = teacher_input_ids[sample_idx]
                matched = False
                for candidate in search_candidates:
                    cand_len = candidate.numel()
                    if cand_len == 0 or cand_len > full_ids.numel():
                        continue
                    for start in range(full_ids.numel() - cand_len, -1, -1):
                        if torch.equal(full_ids[start:start + cand_len], candidate):
                            teacher_mask[sample_idx, start:start + cand_len] = True
                            matched = True
                            break
                    if matched:
                        break

                if not matched and tokenizer is not None:
                    answer_text = tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True).strip()
                    teacher_text = tokenizer.decode(full_ids.tolist(), skip_special_tokens=False)
                    if answer_text and teacher_text:
                        text_start = teacher_text.rfind(answer_text)
                        if text_start != -1:
                            prefix_text = teacher_text[:text_start]
                            prefix_plus_answer_text = teacher_text[: text_start + len(answer_text)]
                            prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
                            prefix_plus_ids = tokenizer(prefix_plus_answer_text, add_special_tokens=False).input_ids
                            if len(prefix_plus_ids) > len(prefix_ids):
                                start = len(prefix_ids)
                                end = len(prefix_plus_ids)
                                if end <= full_ids.numel():
                                    teacher_mask[sample_idx, start:end] = True
                                    matched = True

        return teacher_mask

    raise KeyError(
        "Missing `teacher_answer_mask` in batch. "
        "Extraction requires dataset to return a strict answer-region mask."
    )


def decode_masked_text(tokenizer, input_ids, mask):
    if mask.numel() == 0 or not mask.any():
        return ""
    return tokenizer.decode(input_ids[mask].tolist(), skip_special_tokens=False)


def normalize_answer_text(text):
    text = str(text).strip()
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    text = re.sub(r"\s+", "", text)
    return text


def log_alignment_mismatch(kind, step, sample_idx, layer_idx, teacher_tokens, student_tokens, teacher_text, student_text):
    print(
        f"Warning: {kind} token mismatch at step {step}, sample {sample_idx}, layer {layer_idx}: "
        f"teacher_tokens={teacher_tokens}, student_tokens={student_tokens}"
    )
    print(f"Teacher masked text: {teacher_text}")
    print(f"Student masked text: {student_text}")


def extract_sample_attn_tokens(attn_hs, sample_idx, token_mask, num_heads, head_dim):
    sample_attn = attn_hs[sample_idx]

    if sample_attn.dim() != 3:
        raise ValueError(f"Unexpected sample attention rank: {tuple(sample_attn.shape)}")

    if sample_attn.shape[0] == num_heads and sample_attn.shape[2] == head_dim:
        return sample_attn[:, token_mask, :].permute(1, 0, 2).contiguous()

    if sample_attn.shape[1] == num_heads and sample_attn.shape[2] == head_dim:
        return sample_attn[token_mask].contiguous()

    raise ValueError(
        f"Cannot infer sample attention layout: {tuple(sample_attn.shape)}, "
        f"expected num_heads={num_heads}, head_dim={head_dim}"
    )


ANSWER_SPAN_ERROR_RE = re.compile(
    r"Failed to locate teacher answer span for batch sample (?P<sample_idx>\d+)"
)


def build_language_token_mask(input_ids, attention_mask, vision_token_ids, pad_id, eos_id):
    if attention_mask is None:
        lang_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        lang_mask = attention_mask.bool()

    if vision_token_ids.numel() > 0:
        lang_mask &= ~torch.isin(input_ids, vision_token_ids)

    if pad_id is not None:
        lang_mask &= input_ids.ne(pad_id)

    if eos_id is not None:
        lang_mask &= input_ids.ne(eos_id)

    return lang_mask


def collate_batch_with_bad_sample_retry(collate_fn, batch_samples, batch_indices):
    pending_samples = list(batch_samples)
    pending_indices = list(batch_indices)
    skipped_samples = []

    while pending_samples:
        try:
            return collate_fn(pending_samples), skipped_samples
        except ValueError as exc:
            message = str(exc)
            match = ANSWER_SPAN_ERROR_RE.search(message)
            if match is None:
                raise

            failed_pos = int(match.group("sample_idx"))
            if failed_pos < 0 or failed_pos >= len(pending_samples):
                raise

            failed_sample = pending_samples.pop(failed_pos)
            failed_index = pending_indices.pop(failed_pos)
            failed_pid = failed_sample.get("pid") if isinstance(failed_sample, dict) else None
            failed_answer = None
            if isinstance(failed_sample, dict):
                failed_answer = failed_sample.get("gt_numerical", failed_sample.get("answer"))

            skipped_samples.append(
                {
                    "dataset_index": failed_index,
                    "pid": failed_pid,
                    "answer": failed_answer,
                    "error": message,
                }
            )
            print(
                "Warning: skipping sample with unmatched teacher answer span "
                f"(dataset_index={failed_index}, pid={failed_pid}, answer={failed_answer!r})."
            )

    return None, skipped_samples


def iter_safe_batches(dataloader, seed=None):
    data_source = dataloader.dataset
    collate_fn = dataloader.collate_fn
    batch_size = dataloader.batch_size or 1

    indices = list(range(len(data_source)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_samples = [data_source[idx] for idx in batch_indices]
        batch, skipped_samples = collate_batch_with_bad_sample_retry(
            collate_fn=collate_fn,
            batch_samples=batch_samples,
            batch_indices=batch_indices,
        )
        yield batch, skipped_samples


class TokenizerProcessorAdapter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, *args, **kwargs):
        kwargs.setdefault("add_special_tokens", False)
        return self.tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


@hydra.main(config_path="config", config_name="extract_cot_vector.yaml", version_base=None)
def main(cfg: DictConfig):
    device_idx = getattr(cfg, "devices", 0)
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {cfg.model_name} on {device}")

    model_hf_id, model_path = build_model(cfg)

    is_vl_model = "vl" in str(cfg.model_name).lower()
    if is_vl_model:
        lmm = QwenVLModelWrapper(
            model_root=model_path,
            processor_class=AutoProcessor,
            model_class=Qwen2_5_VLForConditionalGeneration,
            support_models=[model_hf_id],
            local_files_only=True,
            torch_dtype=eval(cfg.dtype),
            processor_args={"padding_side": "left"},
            model_args={"output_hidden_states": True},
            model_name=cfg.model_name,
        ).to(device)

        if hasattr(lmm.processor, "image_processor"):
            lmm.processor.image_processor.max_pixels = 512 * 28 * 28
            lmm.processor.image_processor.min_pixels = 256 * 28 * 28
            print(f"Set max_pixels to {lmm.processor.image_processor.max_pixels} to prevent token mismatch.")
    else:
        lmm = QwenModelWrapper(
            model_root=model_path,
            processor_class=AutoTokenizer,
            model_class=AutoModelForCausalLM,
            support_models=[model_hf_id],
            local_files_only=True,
            torch_dtype=eval(cfg.dtype),
            processor_args={"padding_side": "left", "use_fast": False},
            model_args={"output_hidden_states": True},
            model_name=cfg.model_name,
        ).to(device)
        if lmm.processor.pad_token is None:
            lmm.processor.pad_token = lmm.processor.eos_token
        lmm.processor = TokenizerProcessorAdapter(lmm.processor)

    lmm.eval()
    for param in lmm.model.parameters():
        param.requires_grad = False

    # -----------------------------
    # 构建视觉 token 过滤器
    # -----------------------------
    tok = lmm.processor.tokenizer if hasattr(lmm.processor, "tokenizer") else lmm.processor
    vision_token_ids = []

    if is_vl_model:
        for token_str, token_id in tok.get_vocab().items():
            n = token_str.lower()
            if (
                "image" in n
                or "vision" in n
                or "patch" in n
                or n.startswith("<|im_")
                or n.startswith("<|image_")
            ):
                vision_token_ids.append(token_id)

    vision_token_ids = torch.tensor(sorted(set(vision_token_ids)), device=device, dtype=torch.long)
    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id

    print(f"Identified {len(vision_token_ids)} vision-related token IDs: {vision_token_ids.cpu().tolist()}")

    # -----------------------------
    # 数据集
    # -----------------------------
    cfg.data.training_mode = "TRAIN_STUDENT_DIRECT_Q_SELF_COT"
    dataset = dataset_mapping[cfg.data.name](cfg.data, model_processor=lmm.processor, model_name=cfg.model_name)
    shard_count = int(getattr(cfg, "shard_count", 1) or 1)
    shard_index = int(getattr(cfg, "shard_index", 0) or 0)
    if shard_count > 1:
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError(
                f"Invalid shard_index={shard_index} for shard_count={shard_count}."
            )
        shard_indices = list(range(shard_index, len(dataset._support_set), shard_count))
        dataset._support_set = dataset._support_set.select(shard_indices)
        print(
            f"Extracting shard {shard_index + 1}/{shard_count} with "
            f"{len(dataset._support_set)} support samples."
        )
    dataloader = dataset.train_dataloader(lmm, batch_size=cfg.batch_size)

    # -----------------------------
    # Shift encoder
    # -----------------------------
    shift_encoder = hydra.utils.instantiate(
        cfg.encoder.cls,
        lmm=lmm,
        attn_strategy="ShiftStrategy.VECTOR_SHIFT | ShiftStrategy.RECORD_HIDDEN_STATES | ShiftStrategy.MULTI_HEAD | ShiftStrategy.RECORD_RAW_ATTN_OUTPUTS",
        ffn_strategy="ShiftStrategy.RECORD_HIDDEN_STATES",
        _recursive_=False,
    ).to(device)

    shift_encoder.eval()

    num_layers = lmm.model.config.num_hidden_layers
    hidden_size = lmm.model.config.hidden_size
    num_heads = lmm.model.config.num_attention_heads
    head_dim = hidden_size // num_heads

    _ = shift_encoder.register_shift_hooks()

    # -----------------------------
    # 累加器
    # -----------------------------
    accumulated_ffn_gaps = [
        torch.zeros(hidden_size, device=device) for _ in range(num_layers)
    ]
    accumulated_ffn_hs = [
        torch.zeros(hidden_size, device=device) for _ in range(num_layers)
    ]
    ffn_sample_count_per_layer = [0 for _ in range(num_layers)]

    accumulated_attn_gaps = [
        torch.zeros(num_heads, head_dim, device=device) for _ in range(num_layers)
    ]
    attn_sample_count_per_layer = [0 for _ in range(num_layers)]

    print("Starting CoT vector extraction with answer-region averaging...")

    skipped_sample_count = 0
    dataloader_batch_size = dataloader.batch_size or cfg.batch_size or 1
    total_batches = (len(dataloader.dataset) + dataloader_batch_size - 1) // dataloader_batch_size

    for step, (batch, skipped_samples) in enumerate(
        tqdm(
            iter_safe_batches(dataloader, seed=getattr(cfg.data, "seed", None)),
            total=total_batches,
            desc="Extracting CoT vectors",
        )
    ):
        if skipped_samples:
            skipped_sample_count += len(skipped_samples)
        if batch is None:
            print(f"Warning: skipped empty batch at step {step} after removing invalid samples.")
            continue
        # ============================================================
        # 1) Teacher pass: Q + CoT + A
        # ============================================================
        shift_encoder.reset_hidden_states()
        shift_encoder.current_recording_context = "teacher"
        hooks_teacher = shift_encoder.register_record_hooks()

        with torch.no_grad():
            teacher_inputs = {
                k: v.to(device)
                for k, v in batch["prefix_inputs"].items()
                if isinstance(v, torch.Tensor)
            }
            teacher_inputs.pop("offset_mapping", None)

            _ = lmm.model(**teacher_inputs, output_hidden_states=True)

            teacher_raw_attn_hs = [
                t.clone().to(device) if t is not None else None # 显式搬运到 device
                for t in shift_encoder.teacher_raw_attn_outputs
            ]
            teacher_ffn_hs = [
                t.clone().to(device) if t is not None else None
                for t in shift_encoder.ffn_hidden_states
            ]

        remove_hook_dict(hooks_teacher)

        # ============================================================
        # 2) Student pass: Q + A
        # ============================================================
        shift_encoder.reset_hidden_states()
        shift_encoder.current_recording_context = "student"
        hooks_student = shift_encoder.register_record_hooks()

        with torch.no_grad():
            student_inputs = {
                k: v.to(device)
                for k, v in batch["student_inputs"].items()
                if isinstance(v, torch.Tensor)
            }

            _ = lmm.model(**student_inputs, output_hidden_states=True)

            student_raw_attn_hs = [
                t.clone().to(device) if t is not None else None # 显式搬运到 device
                for t in shift_encoder.student_raw_attn_outputs
            ]
            student_ffn_hs = [
                t.clone().to(device) if t is not None else None # 显式搬运到 device
                for t in shift_encoder.ffn_hidden_states
            ]

        remove_hook_dict(hooks_student)

        # answer mask
        student_answer_mask = build_student_answer_mask(batch=batch, device=device)
        teacher_answer_mask = build_teacher_answer_mask(
            batch=batch,
            device=device,
            tokenizer=lmm.processor.tokenizer,
        )

        teacher_ids = teacher_inputs["input_ids"]
        student_ids = student_inputs["input_ids"]
        t_lang_mask = build_language_token_mask(
            input_ids=teacher_ids,
            attention_mask=teacher_inputs.get("attention_mask"),
            vision_token_ids=vision_token_ids,
            pad_id=pad_id,
            eos_id=eos_id,
        )
        s_lang_mask = build_language_token_mask(
            input_ids=student_ids,
            attention_mask=student_inputs.get("attention_mask"),
            vision_token_ids=vision_token_ids,
            pad_id=pad_id,
            eos_id=eos_id,
        )

        teacher_final_mask = teacher_answer_mask & t_lang_mask
        student_final_mask = student_answer_mask & s_lang_mask
        tok = lmm.processor.tokenizer
        aligned_sample_mask = torch.ones(teacher_ids.shape[0], dtype=torch.bool, device=device)

        for sample_idx in range(teacher_ids.shape[0]):
            teacher_text = decode_masked_text(
                tok,
                teacher_ids[sample_idx].detach().cpu(),
                teacher_final_mask[sample_idx].detach().cpu(),
            )
            student_text = decode_masked_text(
                tok,
                student_ids[sample_idx].detach().cpu(),
                student_final_mask[sample_idx].detach().cpu(),
            )
            if normalize_answer_text(teacher_text) != normalize_answer_text(student_text):
                aligned_sample_mask[sample_idx] = False
                skipped_sample_count += 1
                print(
                    f"Warning: skipping misaligned answer span at step {step}, sample {sample_idx}: "
                    f"teacher='{teacher_text}' student='{student_text}'"
                )

        if step == 0:
            
            teacher_ids_cpu = teacher_ids[0].detach().cpu()
            student_ids_cpu = student_ids[0].detach().cpu()

            teacher_answer_mask_cpu = teacher_answer_mask[0].detach().cpu()
            student_answer_mask_cpu = student_answer_mask[0].detach().cpu()
            teacher_final_mask_cpu = teacher_final_mask[0].detach().cpu()
            student_final_mask_cpu = student_final_mask[0].detach().cpu()

            print("\n===== DEBUG FIRST SAMPLE =====")

            print("\nTeacher FULL TEXT:")
            print(tok.decode(teacher_ids_cpu, skip_special_tokens=False))

            print("\nStudent FULL TEXT:")
            print(tok.decode(student_ids_cpu, skip_special_tokens=False))

            print("\nTeacher answer mask text:")
            print(decode_masked_text(tok, teacher_ids_cpu, teacher_answer_mask_cpu))

            print("\nStudent answer mask text:")
            print(decode_masked_text(tok, student_ids_cpu, student_answer_mask_cpu))

            print("\nTeacher final mask text:")
            print(decode_masked_text(tok, teacher_ids_cpu, teacher_final_mask_cpu))

            print("\nStudent final mask text:")
            print(decode_masked_text(tok, student_ids_cpu, student_final_mask_cpu))

            print("\nTeacher answer token count:", teacher_answer_mask_cpu.sum().item())
            print("Student answer token count:", student_answer_mask_cpu.sum().item())
            print("Teacher final token count:", teacher_final_mask_cpu.sum().item())
            print("Student final token count:", student_final_mask_cpu.sum().item())

            if teacher_final_mask_cpu.sum().item() != student_final_mask_cpu.sum().item():
                print("\nWarning: teacher/student final token counts differ on first sample.")
                print("Teacher final mask text (again):")
                print(decode_masked_text(tok, teacher_ids_cpu, teacher_final_mask_cpu))
                print("Student final mask text (again):")
                print(decode_masked_text(tok, student_ids_cpu, student_final_mask_cpu))

            print("\n=============================\n")

        # ============================================================
        # 4) 逐层提取：answer token 区域平均
        # ============================================================
        batch_size = teacher_ids.shape[0]
        for layer_idx in range(num_layers):
            t_f_hs = teacher_ffn_hs[layer_idx]
            s_f_hs = student_ffn_hs[layer_idx]
            t_a_hs = teacher_raw_attn_hs[layer_idx]
            s_a_hs = student_raw_attn_hs[layer_idx]

            for sample_idx in range(batch_size):
                if not aligned_sample_mask[sample_idx]:
                    continue

                sample_t_mask = teacher_final_mask[sample_idx]
                sample_s_mask = student_final_mask[sample_idx]

                teacher_token_count = int(sample_t_mask.sum().item())
                student_token_count = int(sample_s_mask.sum().item())

                if t_f_hs is not None and s_f_hs is not None:
                    sample_t_ffn = t_f_hs[sample_idx][sample_t_mask]
                    sample_s_ffn = s_f_hs[sample_idx][sample_s_mask]

                    if sample_t_ffn.numel() == 0 or sample_s_ffn.numel() == 0:
                        if teacher_token_count != student_token_count:
                            teacher_text = decode_masked_text(tok, teacher_ids[sample_idx].detach().cpu(), sample_t_mask.detach().cpu())
                            student_text = decode_masked_text(tok, student_ids[sample_idx].detach().cpu(), sample_s_mask.detach().cpu())
                            log_alignment_mismatch(
                                "FFN",
                                step,
                                sample_idx,
                                layer_idx,
                                teacher_token_count,
                                student_token_count,
                                teacher_text,
                                student_text,
                            )
                    elif sample_t_ffn.shape == sample_s_ffn.shape:
                        ffn_gap = sample_t_ffn - sample_s_ffn
                        accumulated_ffn_gaps[layer_idx] += ffn_gap.mean(dim=0)
                        accumulated_ffn_hs[layer_idx] += sample_t_ffn.mean(dim=0)
                        ffn_sample_count_per_layer[layer_idx] += 1
                    else:
                        teacher_text = decode_masked_text(tok, teacher_ids[sample_idx].detach().cpu(), sample_t_mask.detach().cpu())
                        student_text = decode_masked_text(tok, student_ids[sample_idx].detach().cpu(), sample_s_mask.detach().cpu())
                        log_alignment_mismatch(
                            "FFN",
                            step,
                            sample_idx,
                            layer_idx,
                            teacher_token_count,
                            student_token_count,
                            teacher_text,
                            student_text,
                        )

                if t_a_hs is not None and s_a_hs is not None:
                    try:
                        sample_t_attn = extract_sample_attn_tokens(t_a_hs, sample_idx, sample_t_mask, num_heads, head_dim)
                        sample_s_attn = extract_sample_attn_tokens(s_a_hs, sample_idx, sample_s_mask, num_heads, head_dim)

                        if sample_t_attn.numel() == 0 or sample_s_attn.numel() == 0:
                            if teacher_token_count != student_token_count:
                                teacher_text = decode_masked_text(tok, teacher_ids[sample_idx].detach().cpu(), sample_t_mask.detach().cpu())
                                student_text = decode_masked_text(tok, student_ids[sample_idx].detach().cpu(), sample_s_mask.detach().cpu())
                                log_alignment_mismatch(
                                    "Attention",
                                    step,
                                    sample_idx,
                                    layer_idx,
                                    teacher_token_count,
                                    student_token_count,
                                    teacher_text,
                                    student_text,
                                )
                        elif sample_t_attn.shape == sample_s_attn.shape:
                            attn_gap = sample_t_attn - sample_s_attn
                            accumulated_attn_gaps[layer_idx] += attn_gap.mean(dim=0)
                            attn_sample_count_per_layer[layer_idx] += 1
                        else:
                            teacher_text = decode_masked_text(tok, teacher_ids[sample_idx].detach().cpu(), sample_t_mask.detach().cpu())
                            student_text = decode_masked_text(tok, student_ids[sample_idx].detach().cpu(), sample_s_mask.detach().cpu())
                            log_alignment_mismatch(
                                "Attention",
                                step,
                                sample_idx,
                                layer_idx,
                                teacher_token_count,
                                student_token_count,
                                teacher_text,
                                student_text,
                            )

                    except Exception as e:
                        teacher_text = decode_masked_text(tok, teacher_ids[sample_idx].detach().cpu(), sample_t_mask.detach().cpu())
                        student_text = decode_masked_text(tok, student_ids[sample_idx].detach().cpu(), sample_s_mask.detach().cpu())
                        print(
                            f"Warning: attention extraction failed at step {step}, sample {sample_idx}, layer {layer_idx}: {e}"
                        )
                        print(f"Teacher masked text: {teacher_text}")
                        print(f"Student masked text: {student_text}")

        # 显存清理
        del teacher_raw_attn_hs, teacher_ffn_hs, student_raw_attn_hs, student_ffn_hs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Finished extraction with {skipped_sample_count} skipped samples.")

    # ============================================================
    # 5) Final averaging
    # ============================================================
    final_ffn_cot_vector = torch.stack([
        vec / count if count > 0 else torch.zeros_like(vec)
        for vec, count in zip(accumulated_ffn_gaps, ffn_sample_count_per_layer)
    ])

    final_ffn_hs_vector = torch.stack([
        vec / count if count > 0 else torch.zeros_like(vec)
        for vec, count in zip(accumulated_ffn_hs, ffn_sample_count_per_layer)
    ])

    final_attn_cot_vector = torch.stack([
        vec / count if count > 0 else torch.zeros_like(vec)
        for vec, count in zip(accumulated_attn_gaps, attn_sample_count_per_layer)
    ])

    # ============================================================
    # 6) Save
    # ============================================================
    output_dir = os.path.dirname(cfg.output_path)
    os.makedirs(output_dir, exist_ok=True)

    cot_vectors_to_save = {
        "ffn_cot_vector": final_ffn_cot_vector.cpu(),
        "ffn_hs_vector": final_ffn_hs_vector.cpu(),
        "attn_cot_vector": final_attn_cot_vector.cpu() if final_attn_cot_vector is not None else None,
        "ffn_cot_vector_sums": torch.stack(accumulated_ffn_gaps).cpu(),
        "ffn_hs_vector_sums": torch.stack(accumulated_ffn_hs).cpu(),
        "attn_cot_vector_sums": torch.stack(accumulated_attn_gaps).cpu(),
        "ffn_sample_count_per_layer": torch.tensor(ffn_sample_count_per_layer, dtype=torch.long),
        "attn_sample_count_per_layer": torch.tensor(attn_sample_count_per_layer, dtype=torch.long),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "processed_support_samples": len(dataloader.dataset),
        "skipped_sample_count": skipped_sample_count,
        "encoder_type": cfg.encoder.cls,
        "multi_head_attn_strategy": (
            ShiftStrategy.MULTI_HEAD in shift_encoder.attn_strategy
            if isinstance(shift_encoder, AttnApproximator)
            else True
        ),
    }

    torch.save(cot_vectors_to_save, cfg.output_path)
    print(f"Average CoT vectors saved to {cfg.output_path}")


if __name__ == "__main__":
    main()
