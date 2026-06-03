import ast
import enum
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from difflib import SequenceMatcher

from dataset_utils.interface import DatasetBase

try:
    from latex2sympy2 import latex2sympy
except ImportError:  # pragma: no cover - optional dependency
    latex2sympy = None


def find_lcs_token_ids(teacher_tokens, student_tokens):
    matcher = SequenceMatcher(None, teacher_tokens, student_tokens, autojunk=False)
    lcs_tokens = []
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag == "equal":
            lcs_tokens.extend(teacher_tokens[i1:i2])
    return lcs_tokens


def build_new_mask_from_lcs(original_ids, lcs_tokens):
    mask = [False] * len(original_ids)
    lcs_len = len(lcs_tokens)
    if lcs_len == 0:
        return mask

    for i in range(len(original_ids) - lcs_len + 1):
        if original_ids[i:i + lcs_len] == lcs_tokens:
            for j in range(i, i + lcs_len):
                mask[j] = True
            break
    return mask


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


class MathVisionDataset(DatasetBase):
    support_datasets = ["mathvision"]

    def __init__(
        self,
        data_cfg: DictConfig,
        model_processor: Any = None,
        model_name: Optional[str] = None,
    ) -> None:
        super().__init__(data_cfg, model_processor, model_name)
        self.data_cfg = data_cfg
        self.dataset_state = DatasetState.EVAL_BASELINE
        self.data_root = "/data1/wzy/cot-mimic/mathvision"
        for candidate in ["/data/share/MathVision", self.data_root]:
            if os.path.exists(candidate):
                self.data_root = candidate
                break

        source = getattr(data_cfg, "source", "MathLLMs/MathVision")
        cache_dir = getattr(data_cfg, "cache_dir", None)
        hf_name = getattr(data_cfg, "hf_name", None)
        query_split = getattr(data_cfg, "query_split", getattr(data_cfg, "split", "testmini"))
        support_split = getattr(data_cfg, "support_split", query_split)

        self._query_set = load_dataset(source, split=query_split, cache_dir=cache_dir, name=hf_name)
        try:
            self._support_set = load_dataset(source, split=support_split, cache_dir=cache_dir, name=hf_name)
        except Exception as exc:
            print(
                f"Warning: failed to load MathVision support split '{support_split}': {exc}. "
                f"Falling back to query split '{query_split}'."
            )
            self._support_set = self._query_set

        self._self_cot_data = None
        if hasattr(data_cfg, "use_self_cot") and data_cfg.use_self_cot:
            self._load_self_cot_data(data_cfg)

        if hasattr(data_cfg, "num_query_samples") and data_cfg.num_query_samples > 0:
            num = min(data_cfg.num_query_samples, len(self._support_set))
            rng = random.Random(getattr(data_cfg, "seed", None))
            indices = rng.sample(range(len(self._support_set)), num)
            self._support_set = self._support_set.select(indices)
            print(
                f"MathVision initialized: Support set limited to {len(self._support_set)} samples "
                f"with seed={getattr(data_cfg, 'seed', None)}."
            )

        self.direct_answer_system_prompt = (
            "You are a careful visual math assistant. Put your final answer within \\boxed{}."
        )
        self.cot_system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

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
    def _clean_question_text(question: str) -> str:
        text = str(question or "")
        text = re.sub(r"<image\d+>", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _has_literal_options(options: List[str]) -> bool:
        return bool(options) and "".join(str(opt) for opt in options) == "ABCDE"

    @classmethod
    def create_query(cls, problem, shot_type="solution", use_caption=False, use_ocr=False):
        question = cls._clean_question_text(problem.get("question", ""))
        options = problem.get("options") or []

        parts = [f"Question: {question}"]
        if options and not cls._has_literal_options(options):
            option_lines = ["Choices:"]
            for idx, option in enumerate(options):
                option_lines.append(f"({chr(ord('A') + idx)}) {option}")
            parts.append("\n".join(option_lines))

        if options:
            parts.append("Hint: Please answer with the correct option letter, e.g., A, B, C, D, or E.")
        else:
            parts.append("Hint: Please solve the problem and give the final answer at the end.")

        return "\n".join([p for p in parts if p]).strip()

    def _load_image(self, item_or_image):
        if item_or_image is None:
            return None

        if isinstance(item_or_image, dict):
            image = item_or_image.get("decoded_image")
            if isinstance(image, Image.Image):
                return image.convert("RGB")
            image_path = item_or_image.get("image")
        else:
            image = item_or_image
            if isinstance(image, Image.Image):
                return image.convert("RGB")
            image_path = item_or_image if isinstance(item_or_image, str) else None

        if isinstance(image_path, str):
            if os.path.exists(image_path):
                return Image.open(image_path).convert("RGB")
            full_path = os.path.join(self.data_root, image_path)
            if os.path.exists(full_path):
                return Image.open(full_path).convert("RGB")
            full_path = os.path.join(self.data_root, "images", os.path.basename(image_path))
            if os.path.exists(full_path):
                return Image.open(full_path).convert("RGB")
        return None

    def _load_self_cot_data(self, data_cfg: DictConfig):
        path = getattr(data_cfg, "self_cot_path", None)
        if not path or not os.path.exists(path):
            print(f"Warning: Self-CoT path not found at {path}")
            return

        self._self_cot_data = {}
        filtered_indices = []
        for idx, sample in enumerate(self._support_set):
            pass

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                pid = d.get("pid", d.get("id"))
                if pid is not None:
                    self._self_cot_data[str(pid)] = d

        filtered_indices = []
        for idx, sample in enumerate(self._support_set):
            pid = str(sample.get("id"))
            record = self._self_cot_data.get(pid)
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
            print(f"MathVision: Loaded {len(self._support_set)} validated samples from Self-CoT data.")
        else:
            print("Warning: No valid Self-CoT samples found after filtering.")

    def _create_vlm_messages(
        self,
        question: str,
        state: DatasetState,
        item: Dict = None,
        include_image: bool = True,
    ):
        messages = []
        sys_prompt = self.cot_system_prompt if "TEACHER" in state.value else self.direct_answer_system_prompt
        messages.append({"role": "system", "content": [{"type": "text", "text": sys_prompt}]})

        image = self._load_image(item) if (item is not None and include_image) else None
        user_content = []
        if image is not None:
            user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": user_content})

        if "TEACHER" in state.value:
            if item is None:
                raise ValueError("Teacher message requires item with self_cot.")
            record = self._self_cot_data.get(str(item.get("id"))) if self._self_cot_data else None
            cot = record.get("self_cot", "") if record else item.get("self_cot", "")
            if not isinstance(cot, str) or not cot.strip():
                raise ValueError(f"Missing self_cot for id={item.get('id')}")
            ans = record.get("gt_numerical", item.get("answer")) if record else item.get("answer")
            teacher_msg = f"{cot.strip()}\n#### {ans}"
            messages.append({"role": "assistant", "content": [{"type": "text", "text": teacher_msg}]})

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

    @staticmethod
    def _safe_literal_eval(text: str):
        try:
            return ast.literal_eval(text)
        except Exception:
            return None

    @staticmethod
    def _safe_numeric_eval(text: str):
        text = str(text).strip()
        if not text or not re.fullmatch(r"[\d\.\-\+\*/\(\)\s]+", text):
            return None
        try:
            return eval(text, {"__builtins__": {}}, {})
        except Exception:
            return None

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

    @classmethod
    def _is_equal(cls, answer: str, ground_truth: str) -> bool:
        if answer is None or ground_truth is None:
            return False

        asw = str(answer).strip()
        gt = str(ground_truth).strip()
        if not asw or not gt:
            return False
        if asw.lower() == gt.lower():
            return True

        parsed_asw = cls._safe_literal_eval(asw)
        parsed_gt = cls._safe_literal_eval(gt)
        if parsed_asw is not None and parsed_gt is not None and parsed_asw == parsed_gt:
            return True

        num_asw = cls._safe_numeric_eval(asw)
        num_gt = cls._safe_numeric_eval(gt)
        if num_asw is not None and num_gt is not None:
            try:
                if abs(float(num_asw) - float(num_gt)) < 1e-6:
                    return True
            except Exception:
                pass

        if latex2sympy is not None:
            try:
                sym_asw = latex2sympy(asw)
                sym_gt = latex2sympy(gt)
                if abs(float(sym_asw - sym_gt)) < 1e-6:
                    return True
            except Exception:
                pass

        return False

    @staticmethod
    def _normalize_text_for_choice(text: str) -> List[str]:
        normalized = str(text or "")
        for ch in ".()[],:;!*#{}":
            normalized = normalized.replace(ch, " ")
        return [token.strip() for token in normalized.split() if token.strip()]

    @classmethod
    def _can_infer_option(cls, answer: str, choices: Dict[str, str]):
        tokens = cls._normalize_text_for_choice(answer)
        hits = [key for key in choices if key in tokens]
        if len(hits) == 1:
            return hits[0]
        if len(hits) == 0:
            return False
        return False

    @staticmethod
    def _can_infer_text(answer: str, choices: Dict[str, str]):
        lowered = str(answer or "").lower()
        hits = []
        for key, value in choices.items():
            value_text = str(value).strip().lower()
            if value_text and value_text in lowered:
                hits.append(key)
        if len(hits) == 1:
            return hits[0]
        return False

    @classmethod
    def _can_infer_choice(cls, answer: str, choices: Dict[str, str]):
        return cls._can_infer_option(answer, choices) or cls._can_infer_text(answer, choices)

    @staticmethod
    def _list_to_dict(options: List[str]) -> Dict[str, str]:
        return {chr(ord("A") + idx): str(option) for idx, option in enumerate(options)}

    def extract_answer(self, prediction: str) -> str:
        if prediction is None:
            return ""

        text = str(prediction).strip()
        if not text:
            return ""

        answer_tag = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if answer_tag:
            return answer_tag[-1].strip()

        boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
        if boxed:
            return boxed[-1].strip()

        sharp = re.findall(r"####\s*([^\n]+)", text)
        if sharp:
            return sharp[-1].strip()

        final_answer = re.findall(r"(?:Final Answer|Answer|answer)\s*[:：]\s*([^\n]+)", text)
        if final_answer:
            return final_answer[-1].strip()

        choice = re.findall(r"\b([A-E])\b", text)
        if choice:
            return choice[-1].strip()

        fraction = re.findall(r"-?\d+\s*/\s*-?\d+", text)
        if fraction:
            return fraction[-1].replace(" ", "")

        numbers = re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", text, re.IGNORECASE)
        if numbers:
            return numbers[-1].strip()

        return text

    def _resolve_correct_choice(self, answer: str, options: List[str]) -> Optional[str]:
        choices = self._list_to_dict(options)
        normalized = str(answer).strip()
        if normalized in choices:
            return normalized

        upper = normalized.upper()
        if upper in choices:
            return upper

        for key, value in choices.items():
            if self._is_equal(normalized, value) or normalized.lower() == str(value).strip().lower():
                return key
        return None

    def _is_correct(self, item: Dict, extracted_answer: str, raw_prediction: str) -> bool:
        answer = str(item.get("answer", "")).strip()
        options = item.get("options") or []

        if options:
            choices = self._list_to_dict(options)
            correct_choice = self._resolve_correct_choice(answer, options)
            inferred_choice = (
                self._can_infer_choice(raw_prediction, choices)
                or self._can_infer_choice(extracted_answer, choices)
            )

            if correct_choice and inferred_choice:
                return correct_choice == inferred_choice

            extracted_upper = str(extracted_answer).strip().upper()
            if correct_choice and extracted_upper in choices:
                return extracted_upper == correct_choice

            if correct_choice:
                correct_text = choices[correct_choice]
                if self._is_equal(extracted_answer, correct_text) or self._is_equal(raw_prediction, correct_text):
                    return True

        return self._is_equal(extracted_answer, answer) or self._is_equal(raw_prediction, answer)

    def _processor_batch(self, texts: List[str], images: Optional[List[Any]] = None, **kwargs):
        if images is not None and any(img is not None for img in images):
            return self.model_processor(text=texts, images=images, **kwargs)
        return self.model_processor(text=texts, **kwargs)

    def collate_fn_for_train(self, batch_data: List[Dict], max_seq_len: int) -> Dict:
        use_images = bool(getattr(self.data_cfg, "train_use_images", True))
        tokenizer = getattr(self.model_processor, "tokenizer", self.model_processor)
        teacher_prompts, teacher_images, teacher_ans_token_ids = [], [], []
        teacher_plain_ans_token_ids = []
        student_id_list, student_images, student_answer_lengths = [], [], []

        for item in batch_data:
            question = self.create_query(item)
            img = self._load_image(item) if use_images else None
            record = self._self_cot_data.get(str(item.get("id"))) if self._self_cot_data else None
            final_ans = str(record.get("gt_numerical", item.get("answer")) if record else item.get("answer")).strip()

            if self.dataset_state == DatasetState.TRAIN_STUDENT_DIRECT_Q_SELF_COT:
                t_state = DatasetState.TRAIN_TEACHER_SELF_COT
            elif self.dataset_state == DatasetState.TRAIN_STUDENT_DIRECT_Q:
                t_state = DatasetState.TRAIN_TEACHER
            else:
                t_state = DatasetState.TRAIN_TEACHER_SELF_COT if "SELF_COT" in self.dataset_state.value else DatasetState.TRAIN_TEACHER

            t_msgs = self._create_vlm_messages(question, t_state, item=item, include_image=use_images)
            t_text = self._apply_chat_template(t_msgs, add_generation_prompt=False)
            teacher_prompts.append(t_text)
            teacher_images.append(img)

            ans_ids_batch = self.model_processor(text=final_ans, add_special_tokens=False).input_ids
            ans_ids = self._normalize_token_ids(ans_ids_batch)
            if len(ans_ids) == 0:
                raise ValueError(f"Empty answer tokenization for id={item.get('id')}, answer={final_ans!r}")
            teacher_plain_ans_token_ids.append(ans_ids)

            teacher_ctx_ids_batch = self.model_processor(text=" " + final_ans, add_special_tokens=False).input_ids
            teacher_ctx_ids = self._normalize_token_ids(teacher_ctx_ids_batch)
            teacher_ans_token_ids.append(teacher_ctx_ids if len(teacher_ctx_ids) > 0 else ans_ids)

            if self.dataset_state in [DatasetState.TRAIN_STUDENT_DIRECT_Q, DatasetState.TRAIN_STUDENT_DIRECT_Q_SELF_COT]:
                s_state = self.dataset_state
            else:
                s_state = DatasetState.TRAIN_STUDENT_DIRECT_Q_SELF_COT if "SELF_COT" in self.dataset_state.value else DatasetState.TRAIN_STUDENT_DIRECT_Q

            s_msgs = self._create_vlm_messages(question, s_state, item=item, include_image=use_images)
            s_text = self._apply_chat_template(s_msgs, add_generation_prompt=True)
            s_inputs = self._processor_batch([s_text], [img], return_tensors="pt")
            s_p_ids = s_inputs.input_ids[0].tolist()

            student_answer_ids = ans_ids
            s_full_text = s_text + final_ans
            s_full_inputs = self._processor_batch([s_full_text], [img], return_tensors="pt")
            s_full_ids = s_full_inputs.input_ids[0].tolist()
            if s_full_ids[:len(s_p_ids)] == s_p_ids:
                student_answer_ids = s_full_ids[len(s_p_ids):]

            eos_id = [tokenizer.eos_token_id]
            full_s_ids = [int(x) for x in (s_p_ids + student_answer_ids + eos_id)]
            student_id_list.append(torch.tensor(full_s_ids, dtype=torch.long))
            student_images.append(img)
            student_answer_lengths.append(len(student_answer_ids))

        t_batch = self._processor_batch(teacher_prompts, teacher_images, return_tensors="pt", padding=True)
        t_mask = torch.zeros_like(t_batch.input_ids, dtype=torch.bool)
        for i, ans_seq in enumerate(teacher_ans_token_ids):
            full_list = t_batch.input_ids[i].tolist()
            matched = False
            for j in range(len(full_list) - len(ans_seq), -1, -1):
                if full_list[j:j + len(ans_seq)] == ans_seq:
                    t_mask[i, j:j + len(ans_seq)] = True
                    matched = True
                    break
            if not matched and teacher_plain_ans_token_ids[i] != ans_seq:
                plain_ans_seq = teacher_plain_ans_token_ids[i]
                for j in range(len(full_list) - len(plain_ans_seq), -1, -1):
                    if full_list[j:j + len(plain_ans_seq)] == plain_ans_seq:
                        t_mask[i, j:j + len(plain_ans_seq)] = True
                        matched = True
                        break
            if not matched:
                answer_text = str(batch_data[i].get("answer", "")).strip()
                raise ValueError(
                    f"Failed to locate teacher answer span for batch sample {i}, id={batch_data[i].get('id')}, answer={answer_text!r}"
                )

        s_padded = torch.nn.utils.rnn.pad_sequence(
            student_id_list,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
            padding_side="left",
        )

        if s_padded.shape[1] > max_seq_len:
            s_padded = s_padded[:, -max_seq_len:]

        s_labels = torch.full_like(s_padded, -100)
        student_answer_mask = torch.zeros_like(s_padded, dtype=torch.bool)
        for i, ans_len in enumerate(student_answer_lengths):
            answer_start_idx = -ans_len - 1
            answer_end_idx = -1
            s_labels[i, answer_start_idx:answer_end_idx] = s_padded[i, answer_start_idx:answer_end_idx]
            student_answer_mask[i, answer_start_idx:answer_end_idx] = True

        for i in range(len(batch_data)):
            masked_teacher_input = t_batch.input_ids[i][t_mask[i].bool()].tolist()
            masked_student_input = s_padded[i][student_answer_mask[i].bool()].tolist()

            if len(masked_teacher_input) != len(masked_student_input):
                lcs_tokens = find_lcs_token_ids(masked_teacher_input, masked_student_input)
                if len(lcs_tokens) > 0:
                    new_teacher_mask = torch.tensor(
                        build_new_mask_from_lcs(t_batch.input_ids[i].tolist(), lcs_tokens),
                        dtype=torch.bool,
                    )
                    new_student_mask = torch.tensor(
                        build_new_mask_from_lcs(s_padded[i].tolist(), lcs_tokens),
                        dtype=torch.bool,
                    )
                else:
                    min_len = min(len(masked_teacher_input), len(masked_student_input))
                    teacher_keep = t_mask[i].nonzero(as_tuple=True)[0][-min_len:]
                    student_keep = student_answer_mask[i].nonzero(as_tuple=True)[0][-min_len:]
                    new_teacher_mask = torch.zeros_like(t_mask[i], dtype=torch.bool)
                    new_student_mask = torch.zeros_like(student_answer_mask[i], dtype=torch.bool)
                    new_teacher_mask[teacher_keep] = True
                    new_student_mask[student_keep] = True

                t_mask[i] = new_teacher_mask
                student_answer_mask[i] = new_student_mask
                s_labels[i] = torch.full_like(s_labels[i], -100)
                s_labels[i][new_student_mask] = s_padded[i][new_student_mask]

        student_inputs = {
            "input_ids": s_padded,
            "attention_mask": s_padded.ne(tokenizer.pad_token_id).long(),
        }
        if use_images and any(img is not None for img in student_images):
            s_pixels = self.model_processor(text=[""] * len(student_images), images=student_images, return_tensors="pt")
            student_inputs["pixel_values"] = s_pixels.pixel_values
            student_inputs["image_grid_thw"] = s_pixels.image_grid_thw

        return {
            "prefix_inputs": t_batch,
            "teacher_answer_mask": t_mask,
            "student_inputs": student_inputs,
            "student_labels": s_labels,
            "student_answer_mask": student_answer_mask,
        }

    def train_dataloader(self, model: Any, batch_size: int) -> DataLoader:
        mode = getattr(self.data_cfg, "training_mode", "TRAIN_STUDENT_DIRECT_Q")
        self.dataset_state = DatasetState(mode)
        return DataLoader(
            self._support_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: self.collate_fn_for_train(b, getattr(self.data_cfg, "max_seq_len", 2048)),
        )

    def eval(self, eval_cfg: DictConfig, model: Any) -> Tuple[List[Dict], Dict]:
        eval_mode = getattr(eval_cfg, "eval_mode", DatasetState.EVAL_BASELINE.value)
        if eval_mode == DatasetState.EVAL_WITH_COT_VECTOR_DIRECT_Q.value:
            self.dataset_state = DatasetState.EVAL_WITH_COT_VECTOR_DIRECT_Q
        elif eval_mode == DatasetState.EVAL_WITH_COT_VECTOR_ONESHOT.value:
            self.dataset_state = DatasetState.EVAL_WITH_COT_VECTOR_ONESHOT
        else:
            self.dataset_state = DatasetState.EVAL_BASELINE

        loader = DataLoader(self._query_set, batch_size=eval_cfg.batch_size, shuffle=False, collate_fn=lambda x: x)
        records: List[Dict] = []
        correct = 0

        for batch in tqdm(loader, desc="MM Eval MathVision"):
            prompts: List[str] = []
            images: List[Optional[Image.Image]] = []
            queries: List[str] = []

            for item in batch:
                query = self.create_query(item)
                image = self._load_image(item)
                if self.model_name and (
                    "llava-onevision" in self.model_name.lower()
                    or "internvl" in self.model_name.lower()
                ):
                    prompt = query
                else:
                    messages = self._create_vlm_messages(query, self.dataset_state, item=item)
                    prompt = self._apply_chat_template(messages, add_generation_prompt=True)
                prompts.append(prompt)
                images.append(image)
                queries.append(query)

            outputs = model.generate(prompts, images=images, **eval_cfg.generation_args)

            for idx, prediction in enumerate(outputs):
                item = batch[idx]
                raw_extraction = self.extract_answer(prediction)
                is_correct = self._is_correct(item, raw_extraction, prediction)
                if is_correct:
                    correct += 1

                records.append(
                    {
                        "id": item.get("id"),
                        "pid": item.get("id"),
                        "image": item.get("image"),
                        "question": item.get("question", ""),
                        "options": item.get("options", []),
                        "answer": item.get("answer", ""),
                        "subject": item.get("subject", ""),
                        "level": item.get("level"),
                        "query": queries[idx],
                        "response": prediction,
                        "raw_extraction": raw_extraction,
                        "is_correct": is_correct,
                    }
                )

        accuracy = correct / len(records) if records else 0.0
        return records, {"accuracy": accuracy}
