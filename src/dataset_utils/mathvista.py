import re
import sys
from typing import Dict, List, Tuple, Any, Optional
from datasets import load_dataset, Dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import evaluate
import os
from tqdm import tqdm
import random
import enum
import torch
import json
from PIL import Image
from Levenshtein import distance

from dataset_utils.interface import DatasetBase
from difflib import SequenceMatcher
from testbed.data import prepare_dataloader, prepare_input
from testbed.models.model_base import ModelBase

try:
    from evaluation.utils import init_judge_client_or_raise, get_chat_response
except ImportError:
    init_judge_client_or_raise = None
    get_chat_response = None

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
    # New Self-CoT training modes
    TRAIN_TEACHER_SELF_COT = "TRAIN_TEACHER_SELF_COT"
    TRAIN_STUDENT_ONESHOT_SELF_COT = "TRAIN_STUDENT_ONESHOT_SELF_COT"
    TRAIN_STUDENT_DIRECT_Q_SELF_COT = "TRAIN_STUDENT_DIRECT_Q_SELF_COT"
    EVAL_BASELINE = "EVAL_BASELINE"
    EVAL_WITH_COT_VECTOR_ONESHOT = "EVAL_WITH_COT_VECTOR_ONESHOT"
    EVAL_WITH_COT_VECTOR_DIRECT_Q = "EVAL_WITH_COT_VECTOR_DIRECT_Q"

class MathVistaDataset(DatasetBase):
    support_datasets = ["mathvista"]
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _DATA_ROOT_CANDIDATES = [
        "/data/share/datasets/mathvista",
        os.path.join(PROJECT_ROOT, "mathvista"),
        "/data1/wzy/cot-mimic/mathvista",
    ]

    @classmethod
    def resolve_data_root(cls) -> str:
        for candidate in cls._DATA_ROOT_CANDIDATES:
            if os.path.exists(candidate):
                return candidate
        return cls._DATA_ROOT_CANDIDATES[0]

    def __init__(self, data_cfg: DictConfig, model_processor: Any = None, model_name: Optional[str] = None) -> None:
        super().__init__(data_cfg, model_processor, model_name)
        self.data_cfg = data_cfg
        self.data_root = self.resolve_data_root()
        self.TESTMINI_PATH = os.path.join(self.data_root, "data", "testmini-00000-of-00001-725687bf7a18d64b.parquet")
        self.TESTFULL_PATHS = [
            os.path.join(self.data_root, "data", "test-00000-of-00002-6b81bd7f7e2065e6.parquet"),
            os.path.join(self.data_root, "data", "test-00001-of-00002-6a611c71596db30f.parquet"),
        ]
        self.GENERATE_500_PATH = os.path.join(self.data_root, "data", "mathvista_generate.json")
        self.EVALUATE_500_PATH = os.path.join(self.data_root, "data", "mathvista_evaluate.json")
        self.query_split = getattr(data_cfg, "query_split", "testmini")
        self.support_split = getattr(data_cfg, "support_split", self.query_split)
        self._query_set = self._load_split(self.query_split)
        self._support_set = self._load_split(self.support_split)
        self.dataset_state = DatasetState.EVAL_BASELINE

        # Load Self-CoT data if specified  
        self._self_cot_data = None
        self._filtered_support_set = None
        if hasattr(data_cfg, "use_self_cot") and data_cfg.use_self_cot:
            self._load_self_cot_data(data_cfg)

        support_limit = getattr(
            data_cfg,
            "num_support_samples",
            getattr(data_cfg, "num_query_samples", 0),
        )
        if support_limit and support_limit > 0:
            num = min(int(support_limit), len(self._support_set))
            rng = random.Random(getattr(data_cfg, "seed", None))
            indices = rng.sample(range(len(self._support_set)), num)
            self._support_set = self._support_set.select(indices)
            print(
                f"MathVista initialized: support_split={self.support_split}, "
                f"query_split={self.query_split}, support size limited to {len(self._support_set)}."
            )
        else:
            print(
                f"MathVista initialized: support_split={self.support_split} ({len(self._support_set)}), "
                f"query_split={self.query_split} ({len(self._query_set)})."
            )

        self.direct_answer_system_prompt = "You are a helpful and precise assistant for solving math problems with visual context. Put your final answer within \\boxed{}."
        self.cot_system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    def _load_split(self, split_name: str):
        split_key = str(split_name).strip().lower()
        if split_key in {"testmini", "mini"}:
            dataset = load_dataset("parquet", data_files={"split": self.TESTMINI_PATH})["split"]
        elif split_key in {"testfull", "full", "test"}:
            dataset = load_dataset("parquet", data_files={"split": self.TESTFULL_PATHS})["split"]
        elif split_key in {"generate", "generate_500"}:
            dataset = load_dataset("json", data_files={"split": self.GENERATE_500_PATH})["split"]
        elif split_key in {"evaluate", "evaluate_500"}:
            dataset = load_dataset("json", data_files={"split": self.EVALUATE_500_PATH})["split"]
        elif os.path.exists(split_name):
            suffix = os.path.splitext(split_name)[1].lower()
            if suffix == ".json":
                dataset = load_dataset("json", data_files={"split": split_name})["split"]
            elif suffix == ".parquet":
                dataset = load_dataset("parquet", data_files={"split": split_name})["split"]
            else:
                raise ValueError(f"Unsupported MathVista split file format: {split_name}")
        else:
            raise ValueError(
                f"Unsupported MathVista split {split_name!r}. "
                "Use one of: testmini, testfull, generate_500, evaluate_500, or a .json/.parquet path."
            )
        return dataset

    @staticmethod
    def metric_key() -> str:
        return "accuracy"

    @property
    def instruction(self) -> str:
        return ""

    @property
    def num_role_in_round(self) -> int:
        return 2

    def _load_image(self, image_data):
        if image_data is None: return None
        if isinstance(image_data, Image.Image): return image_data.convert('RGB')

        if isinstance(image_data, str):
            # 1. 尝试直接拼接 images 目录
            full_path = os.path.join(self.data_root, image_data)
            
            # 2. 如果不存在，尝试补全 images 目录
            if not os.path.exists(full_path):
                full_path = os.path.join(self.data_root, "images", os.path.basename(image_data))
                
            if os.path.exists(full_path):
                return Image.open(full_path).convert('RGB')
            else:
                print(f"Warning: Image not found at {full_path}")
        return None
    
    def get_most_similar(prediction: str, choices: List[str]):
        prediction = str(prediction)
        distances = [distance(prediction, c) for c in choices]
        return choices[distances.index(min(distances))]

    def safe_equal(prediction, answer):
        try:
            return str(prediction).strip() == str(answer).strip()
        except Exception:
            return False

    def extract_answer(self, prediction: str) -> str:
        if prediction is None:
            return ""

        text = str(prediction).strip()

        # ---------- 1. boxed ----------
        boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
        if boxed:
            return boxed[-1].strip()

        # ---------- 2. #### ----------
        sharp = re.findall(r"####\s*([^\n]+)", text)
        if sharp:
            return sharp[-1].strip()

        # ---------- 3. Answer: ----------
        ans = re.findall(r"(?:Answer|Final Answer|answer)\s*[:：]\s*([^\n]+)", text)
        if ans:
            return ans[-1].strip()

        # ---------- 4. numbers ----------
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", text, re.I)
        if nums:
            return nums[-1]

        # ---------- 5. choice ----------
        choice = re.findall(r"\b([A-D])\b", text)
        if choice:
            return choice[-1]

        return text.strip()


    def _extract_final_answer_from_full_cot(self, full_cot_answer: str) -> str:
    # Extracts just the numerical answer (e.g., "123") from "#### 123" or "\boxed{123}"
        return self.extract_answer(full_cot_answer)
    
    @staticmethod
    def _normalize_extracted_answer(extraction, choices, question_type, answer_type, precision):
        """完全对齐官方 calculate_score.py 逻辑"""
        if question_type == 'multi_choice':
            if not isinstance(extraction, str): extraction = str(extraction)
            extraction = extraction.strip()

            # 提取括号内的字母 (A) -> A
            letter = re.findall(r'\(([a-zA-Z])\)', extraction)
            if len(letter) > 0:
                extraction = letter[0].upper()

            options = [chr(ord('A') + i) for i in range(len(choices))]
            if extraction in options:
                ind = options.index(extraction)
                extraction = choices[ind]
            else:
                # 使用 Levenshtein 距离找最接近的选项内容
                distances = [distance(extraction.lower(), str(choice).lower()) for choice in choices]
                ind = distances.index(min(distances))
                extraction = choices[ind]

        elif answer_type == 'integer':
            try:
                extraction = str(int(float(str(extraction).replace(',', ''))))
            except:
                extraction = None
        elif answer_type == 'float':
            try:
                extraction = str(round(float(str(extraction).replace(',', '')), int(precision)))
            except:
                extraction = None
        return extraction
    
    def normalize_answer(x):
        x = x.strip()
        x = re.sub(r"\\text\{.*?\}", "", x)
        x = re.sub(r"[^\w\.\-eE]", "", x)
        return x

    def _load_self_cot_data(self, data_cfg: DictConfig):
        path = getattr(data_cfg, "self_cot_path", None)
        if not path or not os.path.exists(path):
            print(f"Warning: Self-CoT path not found at {path}")
            return
        
        self._self_cot_data = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line.strip())
                pid = d.get('pid')
                if pid is not None:
                    self._self_cot_data[str(pid)] = d
        
        filtered = []
        for sample in self._support_set:
            pid = str(sample['pid'])
            record = self._self_cot_data.get(pid)
            if not record:
                continue
            if 'is_correct' in record and not record.get('is_correct', False):
                continue
            self_cot = record.get('self_cot', "")
            if not isinstance(self_cot, str) or not self_cot.strip():
                continue
            item = dict(sample)
            item['self_cot'] = self_cot.strip()
            item['gt_numerical'] = str(record.get('gt_numerical', sample.get('answer', "")))
            filtered.append(item)
        
        if filtered:
            self._support_set = Dataset.from_list(filtered)
            print(f"MathVista: Loaded {len(self._support_set)} validated samples from Self-CoT data.")
        else:
            print("Warning: No valid Self-CoT samples found after filtering.")

    
    def _create_vlm_messages(self, question: str, state: DatasetState, item: Dict = None):
        messages = []

        sys_prompt = self.cot_system_prompt if "TEACHER" in state.value else self.direct_answer_system_prompt
        messages.append({"role": "system", "content": [{"type": "text", "text": sys_prompt}]})

        # 当前 query
        img = None
        if item is not None:
            img = self._load_image(item.get("image"))

        user_content = [
            {"type": "image", "image": img},
            {"type": "text", "text": question}
        ]
        messages.append({"role": "user", "content": user_content})

        # teacher content (CoT + answer)
        if "TEACHER" in state.value:
            if item is None:
                raise ValueError("Teacher message requires item with self_cot.")
            cot = item.get('self_cot', "")
            if not isinstance(cot, str) or not cot.strip():
                raise ValueError(f"Missing self_cot for pid={item.get('pid')}")
            ans = item.get('gt_numerical', item.get('answer'))
            teacher_msg = f"{cot.strip()}\n#### {ans}"
            messages.append({"role": "assistant", "content": [{"type": "text", "text": teacher_msg}]})

        return messages

    def collate_fn_for_train(self, batch_data: List[Dict], max_seq_len: int) -> Dict:
        teacher_prompts, teacher_images, teacher_ans_token_ids = [], [], []
        teacher_plain_ans_token_ids = []
        student_id_list, student_images, student_answer_lengths = [], [], []

        for item in batch_data:
            question = MathVistaDataset.create_query(item)
            img = self._load_image(item["image"])
            final_ans = str(item.get("gt_numerical", item.get("answer"))).strip()

            # --- Teacher Side ---
            if self.dataset_state == DatasetState.TRAIN_STUDENT_DIRECT_Q_SELF_COT:
                t_state = DatasetState.TRAIN_TEACHER_SELF_COT
            elif self.dataset_state == DatasetState.TRAIN_STUDENT_DIRECT_Q:
                t_state = DatasetState.TRAIN_TEACHER
            else:
                t_state = DatasetState.TRAIN_TEACHER_SELF_COT if "SELF_COT" in self.dataset_state.value else DatasetState.TRAIN_TEACHER
            t_msgs = self._create_vlm_messages(question, t_state, item=item)
            t_text = self.model_processor.apply_chat_template(t_msgs, tokenize=False, add_generation_prompt=False)
            teacher_prompts.append(t_text)
            teacher_images.append(img)

            ans_ids_batch = self.model_processor(text=final_ans, add_special_tokens=False).input_ids
            ans_ids = ans_ids_batch[0].tolist() if hasattr(ans_ids_batch, "tolist") else ans_ids_batch[0]
            if len(ans_ids) == 0:
                raise ValueError(f"Empty answer tokenization for pid={item.get('pid')}, answer={final_ans!r}")
            teacher_plain_ans_token_ids.append(ans_ids)

            teacher_ctx_ids_batch = self.model_processor(text=" " + final_ans, add_special_tokens=False).input_ids
            teacher_ctx_ids = (
                teacher_ctx_ids_batch[0].tolist()
                if hasattr(teacher_ctx_ids_batch, "tolist")
                else teacher_ctx_ids_batch[0]
            )
            teacher_ans_token_ids.append(teacher_ctx_ids if len(teacher_ctx_ids) > 0 else ans_ids)

            # --- Student Side ---
            if self.dataset_state in [DatasetState.TRAIN_STUDENT_DIRECT_Q, DatasetState.TRAIN_STUDENT_DIRECT_Q_SELF_COT]:
                s_state = self.dataset_state
            else:
                s_state = DatasetState.TRAIN_STUDENT_DIRECT_Q_SELF_COT if "SELF_COT" in self.dataset_state.value else DatasetState.TRAIN_STUDENT_DIRECT_Q
            s_msgs = self._create_vlm_messages(question, s_state, item=item)
            s_text = self.model_processor.apply_chat_template(s_msgs, tokenize=False, add_generation_prompt=True)

            s_inputs = self.model_processor(text=[s_text], images=[img], return_tensors="pt")
            s_p_ids = s_inputs.input_ids[0].tolist()

            student_answer_ids = ans_ids
            s_full_text = s_text + final_ans
            s_full_inputs = self.model_processor(text=[s_full_text], images=[img], return_tensors="pt")
            s_full_ids = s_full_inputs.input_ids[0].tolist()
            if s_full_ids[:len(s_p_ids)] == s_p_ids:
                student_answer_ids = s_full_ids[len(s_p_ids):]

            eos_id = [self.model_processor.tokenizer.eos_token_id]
            full_s_ids = [int(x) for x in (s_p_ids + student_answer_ids + eos_id)]
            student_id_list.append(torch.tensor(full_s_ids, dtype=torch.long))
            student_images.append(img)
            student_answer_lengths.append(len(student_answer_ids))

        # 批量处理 Teacher 输入
        t_batch = self.model_processor(text=teacher_prompts, images=teacher_images, return_tensors="pt", padding=True)
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
                answer_text = str(batch_data[i].get("gt_numerical", batch_data[i].get("answer"))).strip()
                raise ValueError(
                    f"Failed to locate teacher answer span for batch sample {i}, pid={batch_data[i].get('pid')}, answer={answer_text!r}"
                )

        # 处理 Student Padding / Labels / Strict Mask
        s_padded = torch.nn.utils.rnn.pad_sequence(
            student_id_list,
            batch_first=True,
            padding_value=self.model_processor.tokenizer.pad_token_id,
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

        s_pixels = self.model_processor(text=[""] * len(student_images), images=student_images, return_tensors="pt")

        return {
            "prefix_inputs": t_batch,
            "teacher_answer_mask": t_mask,
            "student_inputs": {
                "input_ids": s_padded,
                "pixel_values": s_pixels.pixel_values,
                "image_grid_thw": s_pixels.image_grid_thw,
                "attention_mask": s_padded.ne(self.model_processor.tokenizer.pad_token_id).long(),
            },
            "student_labels": s_labels,
            "student_answer_mask": student_answer_mask,
        }

    def train_dataloader(self, model: Any, batch_size: int) -> DataLoader:
        mode = getattr(self.data_cfg, "training_mode", "TRAIN_STUDENT_DIRECT_Q")
        self.dataset_state = DatasetState(mode)
        return DataLoader(self._support_set, batch_size=batch_size, shuffle=True, 
                          collate_fn=lambda b: self.collate_fn_for_train(b, 2048))

    @staticmethod
    def create_query(problem, shot_type="solution", use_caption=False, use_ocr=False, use_cot=False):
        # 1. 生成 Hint
        question_type = problem.get("question_type")
        answer_type = problem.get("answer_type")
        precision = problem.get("precision", 0)
        unit = problem.get("unit", "")
        
        hint_text = ""
        if use_cot:
            if question_type == "multi_choice":
                hint_text = "Hint: Please reason step by step and provide the final option letter in \\boxed{}, e.g., \\boxed{A}."
            elif answer_type == "integer":
                hint_text = "Hint: Please reason step by step and provide the final integer answer in \\boxed{}, e.g., \\boxed{3}."
            elif answer_type == "float":
                hint_text = "Hint: Please reason step by step and provide the final floating-point answer in \\boxed{}."
            elif answer_type == "list":
                hint_text = "Hint: Please reason step by step and provide the final Python list answer in \\boxed{}, e.g., \\boxed{[1, 2, 3]}."
            else:
                hint_text = "Hint: Please reason step by step and provide the final answer in \\boxed{}."
        else:
            if question_type == "multi_choice":
                hint_text = "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
            elif answer_type == "integer":
                hint_text = "Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end."
            elif answer_type == "float":
                hint_text = "Hint: Please answer the question requiring a floating-point number and provide the final value at the end."
            elif answer_type == "list":
                hint_text = "Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], at the end."

        # 2. 拼接 Question
        question_text = f"Question: {problem.get('question', '')}"
        if unit:
            question_text += f" (Unit: {unit})"

        # 3. 拼接 Choices
        choices_text = ""
        choices = problem.get("choices")
        if choices and isinstance(choices, list):
            texts = ["Choices:"]
            for i, choice in enumerate(choices):
                texts.append(f"({chr(ord('A')+i)}) {choice}")
            choices_text = "\n".join(texts)

        # 4. 拼接 Caption/OCR
        caption_text = f"Image description: {problem['caption']}" if use_caption and problem.get("caption") else ""
        ocr_text = f"Image detected text: {problem['ocr']}" if use_ocr and problem.get("ocr") else ""

        # 5. 组合
        # 注意：这里 MathVista 通常用 'query' 字段作为原始问题，但我们需要重组
        elements = [question_text, choices_text, caption_text, ocr_text, hint_text]
        query = "\n".join([e for e in elements if e])
        return query.strip()

    def verify_extraction(extraction):
        extraction = extraction.strip()
        if extraction == "" or extraction == None:
            return False
        return True

    def create_test_prompt(demo_prompt, query, response):
        demo_prompt = demo_prompt.strip()
        test_prompt = f"{query}\n\n{response}"
        full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
        return full_prompt

    def eval(self, eval_cfg: DictConfig, model: Any) -> Tuple[List[Dict], Dict]:
        eval_mode = getattr(eval_cfg, "eval_mode", None)
        eval_mode_norm = eval_mode.upper() if isinstance(eval_mode, str) else None
        if eval_mode_norm == DatasetState.EVAL_WITH_COT_VECTOR_DIRECT_Q.value:
            self.dataset_state = DatasetState.EVAL_WITH_COT_VECTOR_DIRECT_Q
        elif eval_mode_norm == DatasetState.EVAL_BASELINE.value:
            self.dataset_state = DatasetState.EVAL_BASELINE
        else:
            self.dataset_state = DatasetState.EVAL_BASELINE
        eval_samples = self._query_set
  
        loader = DataLoader(eval_samples, batch_size=eval_cfg.batch_size, shuffle=False, collate_fn=lambda x: x)
        
        records, correct = [], 0
        for batch in tqdm(loader, desc="MM Eval MathVista"):
            prompts, images,queries = [], [], []
            for item in batch:
                img = self._load_image(item.get("image"))
                q = self.create_query(item) 

                # LLaVA-OneVision uses a different chat template; fall back to plain query.
                if self.model_name and (
                    "llava-onevision" in self.model_name.lower()
                    or "internvl" in self.model_name.lower()
                ):
                    p_text = q
                else:
                    msgs = self._create_vlm_messages(q, self.dataset_state, item=item)
                    p_text = self.model_processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                
                prompts.append(p_text)
                images.append(img)
                queries.append(q)

            outputs = model.generate(prompts, images=images, **eval_cfg.generation_args)
            
            for i, pred in enumerate(outputs):
                item = batch[i]

                # 原 boxed / #### / Answer 提取
                raw_ext = self._extract_final_answer_from_full_cot(pred)

                # 归一化
                ext_p = self._normalize_extracted_answer(
                    extraction=raw_ext,
                    choices=item.get('choices', []),
                    question_type=item.get('question_type'),
                    answer_type=item.get('answer_type'),
                    precision=item.get('precision', 0)
                )

                # GT 同样归一化
                gt = str(item.get("answer", ""))
                gt_norm = self._normalize_extracted_answer(
                    gt,
                    item.get('choices', []),
                    item.get('question_type'),
                    item.get('answer_type'),
                    item.get('precision', 0)
                )

                is_c = False if (ext_p is None or gt_norm is None) else str(ext_p).strip() == str(gt_norm).strip()
                
                if is_c:
                    correct += 1

                records.append({
                    "pid": item["pid"],
                    "image": item.get("image"),   # 加这一行
                    "question": item.get("question", ""),
                    "choices": item.get("choices", []),

                    "answer": item.get("answer", ""),
                    "question_type": item.get("question_type", ""),
                    "answer_type": item.get("answer_type", ""),

                    "query": q,
                    "response": pred,   

                    "raw_extraction": raw_ext,
                    "ext_p": ext_p,

                    "gt": gt_norm,
                    "is_correct": is_c
                })
        
        acc = correct / len(records) if records else 0
        return records, {"accuracy": acc}

    
