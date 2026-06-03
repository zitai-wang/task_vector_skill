import enum
import hashlib
import io
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from omegaconf import DictConfig
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_utils.interface import DatasetBase


class DatasetState(enum.Enum):
    TRAIN_TEACHER = "TRAIN_TEACHER"
    TRAIN_STUDENT_ONESHOT = "TRAIN_STUDENT_ONESHOT"
    TRAIN_STUDENT_DIRECT_Q = "TRAIN_STUDENT_DIRECT_Q"
    TRAIN_TEACHER_SELF_COT = "TRAIN_TEACHER_SELF_COT"
    TRAIN_STUDENT_ONESHOT_SELF_COT = "TRAIN_STUDENT_ONESHOT_SELF_COT"
    TRAIN_STUDENT_DIRECT_Q_SELF_COT = "TRAIN_STUDENT_DIRECT_Q_SELF_COT"
    EVAL_BASELINE = "EVAL_BASELINE"
    EVAL_WITH_COT_VECTOR_ONESHOT = "EVAL_WITH_COT_VECTOR_ONESHOT"
    EVAL_WITH_COT_VECTOR_DIRECT_Q = "EVAL_WITH_COT_VECTOR_DIRECT_Q"


class ScienceQADataset(DatasetBase):
    support_datasets = ["scienceqa"]

    def __init__(
        self,
        data_cfg: DictConfig,
        model_processor: Any = None,
        model_name: Optional[str] = None,
    ) -> None:
        super().__init__(data_cfg, model_processor, model_name)
        self.data_cfg = data_cfg
        self.dataset_state = DatasetState.EVAL_BASELINE

        self.source = getattr(data_cfg, "source", "/data/share/ScienceQA")
        self.cache_dir = getattr(
            data_cfg,
            "cache_dir",
            os.environ.get("HF_DATASETS_CACHE"),
        )
        data_dir = os.path.join(self.source, "data")
        self._dataset = load_dataset(
            "parquet",
            data_files={
                "train": os.path.join(data_dir, "train-00000-of-00001-1028f23e353fbe3e.parquet"),
                "validation": os.path.join(data_dir, "validation-00000-of-00001-6c7328ff6c84284c.parquet"),
                "test": os.path.join(data_dir, "test-00000-of-00001-f0e719df791966ff.parquet"),
            },
            cache_dir=self.cache_dir,
        )

        self.query_split = getattr(data_cfg, "query_split", "validation")
        self.support_split = getattr(data_cfg, "support_split", "train")
        available_splits = set(self._dataset.keys())
        if self.query_split not in available_splits:
            raise ValueError(
                f"Invalid ScienceQA query_split={self.query_split!r}. "
                f"Available splits: {sorted(available_splits)}."
            )
        if self.support_split not in available_splits:
            raise ValueError(
                f"Invalid ScienceQA support_split={self.support_split!r}. "
                f"Available splits: {sorted(available_splits)}."
            )

        self._query_set = self._dataset[self.query_split]
        self._support_set = self._dataset[self.support_split]

        self._support_limit = getattr(
            data_cfg,
            "num_support_samples",
            getattr(data_cfg, "num_query_samples", 0),
        )
        self._query_limit = getattr(
            data_cfg,
            "num_eval_samples",
            getattr(data_cfg, "num_query_eval_samples", 0),
        )
        self._self_cot_data = None
        self._self_cot_data_by_pid = None
        if hasattr(data_cfg, "use_self_cot") and data_cfg.use_self_cot:
            self._load_self_cot_data(data_cfg)
        self._apply_support_limit()
        self._apply_query_limit()

        print(
            f"ScienceQA initialized: support_split={self.support_split} ({len(self._support_set)}), "
            f"query_split={self.query_split} ({len(self._query_set)})."
        )

        self.direct_answer_system_prompt = (
            "You are a careful multimodal science question answering assistant. "
            "Answer with the option letter only."
        )
        self.cot_system_prompt = (
            "You are a careful multimodal science question answering assistant. "
            "Please reason step by step, and put your final answer within \\boxed{}."
        )

    @staticmethod
    def metric_key() -> str:
        return "accuracy"

    @property
    def instruction(self) -> str:
        return ""

    @property
    def num_role_in_round(self) -> int:
        return 2

    @staticmethod
    def _normalize_text(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip()).strip().lower()

    @staticmethod
    def _choice_dict(item: Dict) -> Dict[str, str]:
        choices = item.get("choices") or []
        return {chr(ord("A") + idx): str(choice).strip() for idx, choice in enumerate(choices)}

    @classmethod
    def _gold_letter_from_record(cls, record: Dict) -> str:
        explicit = str(record.get("gold_letter", "")).strip().upper()
        if explicit and re.fullmatch(r"[A-J]", explicit):
            return explicit

        choices = record.get("choices") or []
        answer = record.get("answer", record.get("gt_answer", record.get("gt_numerical", "")))
        try:
            answer_idx = int(answer)
            if 0 <= answer_idx < len(choices):
                return chr(ord("A") + answer_idx)
        except Exception:
            pass

        answer_text = str(answer or "").strip().upper()
        if re.fullmatch(r"[A-J]", answer_text):
            return answer_text
        return ""

    @classmethod
    def _gold_letter(cls, item: Dict) -> str:
        choices = cls._choice_dict(item)
        if not choices:
            return ""

        answer = item.get("answer")
        try:
            answer_idx = int(answer)
            if 0 <= answer_idx < len(choices):
                return chr(ord("A") + answer_idx)
        except Exception:
            pass

        answer_text = str(answer or "").strip()
        upper = answer_text.upper()
        if upper in choices:
            return upper

        normalized = cls._normalize_text(answer_text)
        for letter, choice_text in choices.items():
            if normalized == cls._normalize_text(choice_text):
                return letter
        return ""

    @classmethod
    def _sample_key_from_fields(
        cls,
        question: Any,
        hint: Any,
        answer_letter: Any,
        subject: Any,
        topic: Any,
        choices: Any = None,
        lecture: Any = None,
        solution: Any = None,
        image_signature: Any = None,
    ) -> str:
        raw_question = str(question or "").strip()
        raw_hint = str(hint or "").strip()

        if "Question:" in raw_question:
            parsed_question = re.search(
                r"Question:\s*(.*?)(?:\nChoices:|\nHint:|$)",
                raw_question,
                flags=re.DOTALL,
            )
            if parsed_question:
                raw_question = parsed_question.group(1).strip()

            if not raw_hint:
                parsed_hint = re.search(
                    r"Context:\s*(.*?)(?:\nQuestion:|$)",
                    str(question or ""),
                    flags=re.DOTALL,
                )
                if parsed_hint:
                    raw_hint = parsed_hint.group(1).strip()

        return "||".join(
            [
                cls._normalize_text(raw_question),
                cls._normalize_text(raw_hint),
                json.dumps(
                    [cls._normalize_text(choice) for choice in (choices or [])],
                    ensure_ascii=False,
                ),
                cls._normalize_text(answer_letter),
                cls._normalize_text(subject),
                cls._normalize_text(topic),
                cls._normalize_text(lecture),
                cls._normalize_text(solution),
                cls._normalize_text(image_signature),
            ]
        )

    @classmethod
    def _sample_key(cls, item: Dict) -> str:
        return cls._sample_key_from_fields(
            item.get("question", ""),
            item.get("hint", ""),
            cls._gold_letter(item),
            item.get("subject", ""),
            item.get("topic", ""),
            item.get("choices", []),
            item.get("lecture", ""),
            item.get("solution", ""),
            cls._image_signature(item.get("image")),
        )

    @staticmethod
    def _image_path(image_data: Any) -> Optional[str]:
        if isinstance(image_data, dict):
            path = image_data.get("path")
            return str(path) if path else None
        if isinstance(image_data, str):
            return image_data
        return None

    @classmethod
    def _image_signature(cls, image_data: Any) -> str:
        if isinstance(image_data, dict):
            image_bytes = image_data.get("bytes")
            if image_bytes:
                return hashlib.sha1(image_bytes).hexdigest()

            path = image_data.get("path")
            if path:
                return cls._normalize_text(path)

        if isinstance(image_data, str):
            return cls._normalize_text(image_data)

        return ""

    def _decode_image(self, image_data: Any) -> Optional[Image.Image]:
        if image_data is None:
            return None
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        if isinstance(image_data, str):
            if os.path.exists(image_data):
                return Image.open(image_data).convert("RGB")
            return None
        if isinstance(image_data, dict):
            path = image_data.get("path")
            if path:
                candidate_paths = [path]
                if not os.path.isabs(path):
                    candidate_paths.append(os.path.join(self.source, path))
                for candidate in candidate_paths:
                    if candidate and os.path.exists(candidate):
                        return Image.open(candidate).convert("RGB")

            image_bytes = image_data.get("bytes")
            if image_bytes:
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return None

    @staticmethod
    def _strip_prediction_wrappers(text: str) -> str:
        candidate = str(text or "").strip()
        if not candidate:
            return ""

        answer_tag = re.findall(r"<answer>(.*?)</answer>", candidate, re.DOTALL | re.IGNORECASE)
        if answer_tag:
            return answer_tag[-1].strip()

        boxed = re.findall(r"\\boxed\{([^}]*)\}", candidate)
        if boxed:
            return boxed[-1].strip()

        explicit_answer = re.findall(
            r"(?:Final Answer|Answer|answer)\s*[:：]\s*([^\n]+)",
            candidate,
        )
        if explicit_answer:
            return explicit_answer[-1].strip()

        return candidate

    def _load_self_cot_data(self, data_cfg: DictConfig):
        path = getattr(data_cfg, "self_cot_path", None)
        if not path or not os.path.exists(path):
            print(f"Warning: ScienceQA Self-CoT path not found at {path}")
            return

        self._self_cot_data = {}
        self._self_cot_data_by_pid = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record_pid = record.get("pid")
                if record_pid not in [None, "", "N/A"]:
                    self._self_cot_data_by_pid[str(record_pid)] = record
                key = self._sample_key_from_fields(
                    record.get("question", ""),
                    record.get("hint", ""),
                    self._gold_letter_from_record(record),
                    record.get("subject", ""),
                    record.get("topic", ""),
                    record.get("choices", []),
                    record.get("lecture", ""),
                    record.get("solution", ""),
                    record.get("image_signature", ""),
                )
                if key.strip("|"):
                    self._self_cot_data[key] = record

        filtered_indices = []
        for idx, item in enumerate(self._support_set):
            record = self._lookup_self_cot_record(item)
            if not record:
                continue
            if "is_correct" in record and not record.get("is_correct", False):
                continue
            self_cot = record.get("self_cot", "")
            if not isinstance(self_cot, str) or not self_cot.strip():
                continue
            filtered_indices.append(idx)

        if filtered_indices:
            self._support_set = self._support_set.select(filtered_indices)
            print(
                f"ScienceQA: Loaded {len(self._support_set)} validated samples from Self-CoT data."
            )
        else:
            print("Warning: No valid ScienceQA Self-CoT samples found after filtering.")

    def _apply_support_limit(self) -> None:
        support_limit = self._support_limit
        if not support_limit or support_limit <= 0:
            return

        num = min(int(support_limit), len(self._support_set))
        if num >= len(self._support_set):
            return

        rng = random.Random(getattr(self.data_cfg, "seed", None))
        indices = rng.sample(range(len(self._support_set)), num)
        self._support_set = self._support_set.select(indices)

    def _apply_query_limit(self) -> None:
        query_limit = self._query_limit
        if not query_limit or query_limit <= 0:
            return

        num = min(int(query_limit), len(self._query_set))
        if num >= len(self._query_set):
            return

        rng = random.Random(getattr(self.data_cfg, "seed", None))
        indices = rng.sample(range(len(self._query_set)), num)
        self._query_set = self._query_set.select(indices)

    def extract_answer(self, prediction: str) -> str:
        if prediction is None:
            return ""

        stripped = self._strip_prediction_wrappers(prediction).strip()
        if not stripped:
            return ""

        bracketed = re.findall(r"\(([A-J])\)", stripped, flags=re.IGNORECASE)
        if bracketed:
            return bracketed[-1].upper()

        direct_letter = re.findall(r"\b([A-J])\b", stripped, flags=re.IGNORECASE)
        if direct_letter:
            return direct_letter[-1].upper()

        numbers = re.findall(r"-?\d+(?:\.\d+)?", stripped)
        if numbers:
            return numbers[-1]

        return stripped

    @classmethod
    def create_query(cls, problem: Dict, answer_mode: str = "letter_only") -> str:
        question = str(problem.get("question", "")).strip()
        hint = str(problem.get("hint", "")).strip()
        choices = problem.get("choices") or []

        parts = []
        if hint:
            parts.append(f"Context: {hint}")
        parts.append(f"Question: {question}")
        if choices:
            option_lines = ["Choices:"]
            for idx, choice in enumerate(choices):
                option_lines.append(f"({chr(ord('A') + idx)}) {choice}")
            parts.append("\n".join(option_lines))

        if answer_mode == "cot":
            parts.append(
                "Hint: Reason step by step and then give the final option letter in \\boxed{}, "
                "for example \\boxed{A}."
            )
        else:
            parts.append("Hint: Answer with the option letter only, for example A, B, C, or D.")

        return "\n".join(part for part in parts if part).strip()

    @classmethod
    def create_cot_query(cls, problem: Dict) -> str:
        return cls.create_query(problem, answer_mode="cot")

    def _build_teacher_rationale(self, item: Dict, state: DatasetState) -> str:
        gold_letter = self._gold_letter(item)

        if "SELF_COT" in state.value and self._self_cot_data is not None:
            record = self._lookup_self_cot_record(item)
            if record is not None:
                cot = str(record.get("self_cot", "")).strip()
                if cot:
                    return f"{cot}\n#### {gold_letter}" if gold_letter else cot

        rationale_parts = []
        lecture = str(item.get("lecture", "")).strip()
        solution = str(item.get("solution", "")).strip()
        if lecture:
            rationale_parts.append(lecture)
        if solution:
            rationale_parts.append(solution)
        rationale_parts.append(f"#### {gold_letter}" if gold_letter else "####")
        return "\n\n".join(rationale_parts)

    def _lookup_self_cot_record(self, item: Dict) -> Optional[Dict]:
        item_pid = item.get("id", item.get("pid"))
        if self._self_cot_data_by_pid is not None and item_pid not in [None, "", "N/A"]:
            matched = self._self_cot_data_by_pid.get(str(item_pid))
            if matched is not None:
                return matched

        if self._self_cot_data is None:
            return None
        return self._self_cot_data.get(self._sample_key(item))

    @staticmethod
    def _normalize_token_ids(token_ids: Any) -> List[int]:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if token_ids is None:
            return []
        if isinstance(token_ids, int):
            return [int(token_ids)]
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return [int(x) for x in token_ids]

    def _create_vlm_messages(
        self,
        question: str,
        state: DatasetState,
        item: Optional[Dict] = None,
        include_image: bool = True,
    ) -> List[Dict]:
        messages = []
        sys_prompt = self.cot_system_prompt if "TEACHER" in state.value else self.direct_answer_system_prompt
        messages.append({"role": "system", "content": [{"type": "text", "text": sys_prompt}]})

        user_content = []
        image = self._decode_image(item.get("image")) if include_image and item is not None else None
        if image is not None:
            user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": user_content})

        if "TEACHER" in state.value and item is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self._build_teacher_rationale(item, state)}],
                }
            )

        return messages

    @staticmethod
    def _flatten_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return "" if content is None else str(content)

        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = str(part.get("text", "")).strip()
                if text:
                    text_parts.append(text)
        return "\n".join(text_parts).strip()

    def _apply_chat_template(self, messages: List[Dict], add_generation_prompt: bool) -> str:
        try:
            return self.model_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except TypeError as exc:
            if "can only concatenate str (not \"list\") to str" not in str(exc):
                raise

            flattened_messages = []
            for message in messages:
                flattened_messages.append(
                    {
                        "role": message.get("role", "user"),
                        "content": self._flatten_message_content(message.get("content")),
                    }
                )

            return self.model_processor.apply_chat_template(
                flattened_messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    def _processor_batch(
        self,
        texts: List[str],
        images: Optional[List[Optional[Image.Image]]] = None,
        padding: bool = True,
    ):
        if images is None:
            return self.model_processor(text=texts, return_tensors="pt", padding=padding)

        has_images = [img is not None for img in images]
        if not any(has_images):
            return self.model_processor(text=texts, return_tensors="pt", padding=padding)

        if any(has_images) and not all(has_images):
            if len(texts) != 1:
                raise ValueError(
                    "ScienceQA mixed image/no-image train batches are only supported with batch_size=1."
                )
            present_idx = has_images.index(True)
            return self.model_processor(
                text=[texts[present_idx]],
                images=[images[present_idx]],
                return_tensors="pt",
                padding=padding,
            )

        return self.model_processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=padding,
        )

    def _resolve_teacher_state(self) -> DatasetState:
        if self.dataset_state == DatasetState.TRAIN_STUDENT_DIRECT_Q_SELF_COT:
            return DatasetState.TRAIN_TEACHER_SELF_COT
        if self.dataset_state == DatasetState.TRAIN_STUDENT_DIRECT_Q:
            return DatasetState.TRAIN_TEACHER
        if "SELF_COT" in self.dataset_state.value:
            return DatasetState.TRAIN_TEACHER_SELF_COT
        return DatasetState.TRAIN_TEACHER

    def _resolve_student_state(self) -> DatasetState:
        if self.dataset_state in [
            DatasetState.TRAIN_STUDENT_DIRECT_Q,
            DatasetState.TRAIN_STUDENT_DIRECT_Q_SELF_COT,
        ]:
            return self.dataset_state
        if "SELF_COT" in self.dataset_state.value:
            return DatasetState.TRAIN_STUDENT_DIRECT_Q_SELF_COT
        return DatasetState.TRAIN_STUDENT_DIRECT_Q

    def collate_fn_for_train(self, batch_data: List[Dict], max_seq_len: int) -> Dict:
        use_images = bool(getattr(self.data_cfg, "train_use_images", True))
        teacher_prompts, teacher_images, teacher_answer_token_ids = [], [], []
        teacher_plain_answer_token_ids = []
        student_id_list, student_images, student_answer_lengths = [], [], []

        teacher_state = self._resolve_teacher_state()
        student_state = self._resolve_student_state()

        for item in batch_data:
            teacher_question = self.create_cot_query(item)
            student_question = self.create_query(item)
            image = self._decode_image(item.get("image")) if use_images else None
            final_answer = self._gold_letter(item)
            if not final_answer:
                raise ValueError(
                    f"Failed to resolve ScienceQA gold letter for question={item.get('question', '')!r}"
                )

            teacher_messages = self._create_vlm_messages(
                teacher_question,
                teacher_state,
                item=item,
                include_image=use_images,
            )
            teacher_text = self._apply_chat_template(
                teacher_messages,
                add_generation_prompt=False,
            )
            teacher_prompts.append(teacher_text)
            teacher_images.append(image)

            answer_ids_batch = self.model_processor(text=final_answer, add_special_tokens=False).input_ids
            answer_ids = self._normalize_token_ids(answer_ids_batch)
            if not answer_ids:
                raise ValueError(
                    f"Empty answer tokenization for ScienceQA answer={final_answer!r}"
                )
            teacher_plain_answer_token_ids.append(answer_ids)

            teacher_ctx_ids_batch = self.model_processor(
                text=" " + final_answer,
                add_special_tokens=False,
            ).input_ids
            teacher_ctx_ids = self._normalize_token_ids(teacher_ctx_ids_batch)
            teacher_answer_token_ids.append(
                teacher_ctx_ids if len(teacher_ctx_ids) > 0 else answer_ids
            )

            student_messages = self._create_vlm_messages(
                student_question,
                student_state,
                item=item,
                include_image=use_images,
            )
            student_text = self._apply_chat_template(
                student_messages,
                add_generation_prompt=True,
            )

            student_prompt_inputs = self._processor_batch([student_text], [image])
            student_prompt_ids = student_prompt_inputs.input_ids[0].tolist()

            student_answer_ids = answer_ids
            full_student_text = student_text + final_answer
            full_student_inputs = self._processor_batch([full_student_text], [image])
            full_student_ids = full_student_inputs.input_ids[0].tolist()
            if full_student_ids[: len(student_prompt_ids)] == student_prompt_ids:
                student_answer_ids = full_student_ids[len(student_prompt_ids) :]

            tokenizer = getattr(self.model_processor, "tokenizer", self.model_processor)
            eos_id = [tokenizer.eos_token_id]
            full_student_sequence = [
                int(token_id)
                for token_id in (student_prompt_ids + student_answer_ids + eos_id)
            ]
            student_id_list.append(torch.tensor(full_student_sequence, dtype=torch.long))
            student_images.append(image)
            student_answer_lengths.append(len(student_answer_ids))

        teacher_batch = self._processor_batch(teacher_prompts, teacher_images, padding=True)
        teacher_answer_mask = torch.zeros_like(teacher_batch.input_ids, dtype=torch.bool)
        for idx, answer_seq in enumerate(teacher_answer_token_ids):
            full_ids = teacher_batch.input_ids[idx].tolist()
            matched = False
            for start in range(len(full_ids) - len(answer_seq), -1, -1):
                if full_ids[start : start + len(answer_seq)] == answer_seq:
                    teacher_answer_mask[idx, start : start + len(answer_seq)] = True
                    matched = True
                    break

            if not matched and teacher_plain_answer_token_ids[idx] != answer_seq:
                plain_seq = teacher_plain_answer_token_ids[idx]
                for start in range(len(full_ids) - len(plain_seq), -1, -1):
                    if full_ids[start : start + len(plain_seq)] == plain_seq:
                        teacher_answer_mask[idx, start : start + len(plain_seq)] = True
                        matched = True
                        break

            if not matched:
                raise ValueError(
                    f"Failed to locate teacher answer span for batch sample {idx}, "
                    f"answer={self._gold_letter(batch_data[idx])!r}"
                )

        tokenizer = getattr(self.model_processor, "tokenizer", self.model_processor)
        student_padded = torch.nn.utils.rnn.pad_sequence(
            student_id_list,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
            padding_side="left",
        )
        if student_padded.shape[1] > max_seq_len:
            student_padded = student_padded[:, -max_seq_len:]

        student_labels = torch.full_like(student_padded, -100)
        student_answer_mask = torch.zeros_like(student_padded, dtype=torch.bool)
        for idx, answer_len in enumerate(student_answer_lengths):
            answer_start_idx = -answer_len - 1
            answer_end_idx = -1
            student_labels[idx, answer_start_idx:answer_end_idx] = student_padded[
                idx, answer_start_idx:answer_end_idx
            ]
            student_answer_mask[idx, answer_start_idx:answer_end_idx] = True

        student_inputs = {
            "input_ids": student_padded,
            "attention_mask": student_padded.ne(tokenizer.pad_token_id).long(),
        }

        if any(img is not None for img in student_images):
            student_pixels = self._processor_batch(
                [""] * len(student_images),
                student_images,
                padding=True,
            )
            if hasattr(student_pixels, "pixel_values"):
                student_inputs["pixel_values"] = student_pixels.pixel_values
            if hasattr(student_pixels, "image_grid_thw"):
                student_inputs["image_grid_thw"] = student_pixels.image_grid_thw

        return {
            "prefix_inputs": teacher_batch,
            "teacher_answer_mask": teacher_answer_mask,
            "student_inputs": student_inputs,
            "student_labels": student_labels,
            "student_answer_mask": student_answer_mask,
        }

    def train_dataloader(self, model: Any, batch_size: int) -> DataLoader:
        del model
        mode = getattr(self.data_cfg, "training_mode", "TRAIN_STUDENT_DIRECT_Q")
        self.dataset_state = DatasetState(mode)
        return DataLoader(
            self._support_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: self.collate_fn_for_train(
                batch,
                getattr(self.data_cfg, "max_seq_len", 4096),
            ),
        )

    def _parse_prediction(self, item: Dict, prediction: str) -> str:
        choices = self._choice_dict(item)
        if not choices:
            return self.extract_answer(prediction)

        extracted = self.extract_answer(prediction).strip()
        if extracted.upper() in choices:
            return extracted.upper()

        search_text = f" {str(prediction or '').strip()} "
        candidates: List[Tuple[int, str]] = []
        for letter in choices:
            patterns = [
                rf"\({letter}\)",
                rf"\b{letter}\b",
                rf"{letter}\.",
                rf"{letter}\)",
            ]
            for pattern in patterns:
                for match in re.finditer(pattern, search_text, flags=re.IGNORECASE):
                    candidates.append((match.start(), letter))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]

        number_matches = re.findall(r"-?\d+(?:\.\d+)?", extracted or search_text)
        if number_matches:
            try:
                numeric = int(float(number_matches[-1]))
                if 0 <= numeric < len(choices):
                    return chr(ord("A") + numeric)
                if 1 <= numeric <= len(choices):
                    return chr(ord("A") + numeric - 1)
            except Exception:
                pass

        normalized_extracted = self._normalize_text(extracted)
        for letter, choice_text in choices.items():
            normalized_choice = self._normalize_text(choice_text)
            if normalized_extracted and normalized_extracted == normalized_choice:
                return letter

        prediction_lower = str(prediction or "").lower()
        text_matches: List[Tuple[int, str]] = []
        for letter, choice_text in choices.items():
            choice_lower = str(choice_text).strip().lower()
            if choice_lower and choice_lower in prediction_lower:
                text_matches.append((prediction_lower.rfind(choice_lower), letter))
        if text_matches:
            text_matches.sort(key=lambda x: x[0])
            return text_matches[-1][1]

        return extracted.upper() if extracted.upper() in choices else ""

    def _is_correct(
        self,
        item: Dict,
        extracted_answer: Optional[str] = None,
        raw_prediction: Optional[str] = None,
    ) -> bool:
        gold_letter = self._gold_letter(item)
        if not gold_letter:
            return False

        if raw_prediction is not None:
            parsed = self._parse_prediction(item, raw_prediction)
        elif extracted_answer is not None:
            parsed = self._parse_prediction(item, extracted_answer)
        else:
            return False
        return parsed == gold_letter

    def eval(self, eval_cfg: DictConfig, model: Any) -> Tuple[List[Dict], Dict]:
        eval_mode = getattr(eval_cfg, "eval_mode", DatasetState.EVAL_BASELINE.value)
        if eval_mode == DatasetState.EVAL_WITH_COT_VECTOR_DIRECT_Q.value:
            self.dataset_state = DatasetState.EVAL_WITH_COT_VECTOR_DIRECT_Q
        elif eval_mode == DatasetState.EVAL_WITH_COT_VECTOR_ONESHOT.value:
            self.dataset_state = DatasetState.EVAL_WITH_COT_VECTOR_ONESHOT
        else:
            self.dataset_state = DatasetState.EVAL_BASELINE

        loader = DataLoader(
            self._query_set,
            batch_size=eval_cfg.batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
        )

        records: List[Dict] = []
        correct = 0

        for batch in tqdm(loader, desc="MM Eval ScienceQA"):
            prompts: List[str] = []
            images: List[Optional[Image.Image]] = []
            queries: List[str] = []

            for item in batch:
                query = self.create_query(item)
                image = self._decode_image(item.get("image"))

                if self.model_processor is None or (
                    self.model_name and "llava-onevision" in self.model_name.lower()
                ):
                    prompt = query
                else:
                    messages = self._create_vlm_messages(query, self.dataset_state, item=item)
                    prompt = self._apply_chat_template(messages, add_generation_prompt=True)

                prompts.append(prompt)
                images.append(image)
                queries.append(query)

            if any(image is not None for image in images):
                outputs = model.generate(prompts, images=images, **eval_cfg.generation_args)
            else:
                outputs = model.generate(prompts, **eval_cfg.generation_args)

            for idx, prediction in enumerate(outputs):
                item = batch[idx]
                parsed_pred = self._parse_prediction(item, prediction)
                gold_letter = self._gold_letter(item)
                is_correct = bool(gold_letter) and parsed_pred == gold_letter
                if is_correct:
                    correct += 1

                records.append(
                    {
                        "pid": item.get("pid", item.get("id", len(records))),
                        "question": item.get("question", ""),
                        "hint": item.get("hint", ""),
                        "image_path": self._image_path(item.get("image")),
                        "subject": item.get("subject", ""),
                        "topic": item.get("topic", ""),
                        "grade": item.get("grade", ""),
                        "task": item.get("task", ""),
                        "category": item.get("category", ""),
                        "skill": item.get("skill", ""),
                        "choices": item.get("choices", []),
                        "answer": item.get("answer", ""),
                        "gold_letter": gold_letter,
                        "sample_key": self._sample_key(item),
                        "query": queries[idx],
                        "response": prediction,
                        "raw_extraction": self.extract_answer(prediction),
                        "parsed_pred": parsed_pred,
                        "is_correct": is_correct,
                    }
                )

        accuracy = correct / len(records) if records else 0.0
        return records, {"accuracy": accuracy}
