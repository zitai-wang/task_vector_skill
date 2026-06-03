import argparse
import json
import re
from fractions import Fraction
from typing import Optional, Tuple

class FractionNormalizer:
    # 预编译正则：匹配三种分数写法
    _RE_SLASH   = re.compile(r"^\s*(-?\d+)\s*/\s*(-?\d+)\s*$")            # 5/12
    _RE_FRAC    = re.compile(r"^\s*\\frac\s*\{\s*(-?\d+)\s*\}\s*\{\s*(-?\d+)\s*\}\s*$")  # \frac{5}{12}
    _RE_MIXED   = re.compile(r"^\s*(-?\d+)\s+(-?\d+)\s*/\s*(-?\d+)\s*$")  # 1 5/12
    # 原有正则不变，再补一条百分数
    _RE_PERCENT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*\\?\%\s*$")  # 20% 或 20\%
    # 1. 根式正则
    _RE_SQRT_UNICODE = re.compile(r"^\s*√(\d+)\s*$")  # √65
    _RE_SQRT_LATEX = re.compile(r"^\s*\\sqrt\s*\{\s*(\d+)\s*\}\s*$")  # \sqrt{65}

    @classmethod
    def parse(cls, raw: str) -> Optional:
        """把任意分数格式转成 Fraction;不是分数返回 None。"""
        raw = raw.strip()
        if not raw:
            return None

        # 根式
        m = cls._RE_SQRT_UNICODE.match(raw) or cls._RE_SQRT_LATEX.match(raw)
        if m:
            return int(m.group(1))

        # 0. 百分数优先处理
        m = cls._RE_PERCENT.match(raw)
        if m:
            return Fraction(float(m.group(1)) / 100).limit_denominator()

        # 1. 匹配 \frac{a}{b}
        m = cls._RE_FRAC.match(raw)
        if m:
            a, b = map(int, m.groups())
            try:
                return Fraction(a, b)
            except ZeroDivisionError:
                return None  # 修复了 \frac{0}{0} 的问题

        # 2. 匹配 a/b
        m = cls._RE_SLASH.match(raw)
        if m:
            a, b = map(int, m.groups())
            try:
                return Fraction(a, b)
            except ZeroDivisionError:
                return None  # 修复了 0/0 的问题

        # 3. 匹配带分数 c a/b
        m = cls._RE_MIXED.match(raw)
        if m:
            c, a, b = map(int, m.groups())
            try:
                return Fraction(c * b + a, b)
            except ZeroDivisionError:
                return None  # 修复了 1 0/0 的问题

        return None


    @staticmethod
    def to_slash(frac: Fraction) -> str:
        """Fraction -> '5/12'"""
        return f"{frac.numerator}/{frac.denominator}"

    @staticmethod
    def to_latex(frac: Fraction) -> str:
        """Fraction -> '\\frac{5}{12}'"""
        return f"\\frac{{{frac.numerator}}}{{{frac.denominator}}}"

    @staticmethod
    def to_percent(frac: Fraction, latex: bool = False) -> str:
        """Fraction -> '20%' 或 '20\%'"""
        val = float(frac) * 100
        if val == int(val):
            s = str(int(val))
        else:
            s = f"{val:g}"  # 去掉多余 .0
        return s + (r"\%" if latex else "%")


    @classmethod
    def is_equal(cls, raw1: str, raw2: str) -> bool:
        """两个字符串分数是否数值相等"""
        f1 = cls.parse(raw1)
        f2 = cls.parse(raw2)
        return f1 is not None and f1 == f2


# file_path = "/home/gavinqi/home/gavinqi/yongliang/project/cot-mimic/results/record/icl-llama-3.1-8b-instruct-gsm8k-0shot-direct/0shot.json"

parser = argparse.ArgumentParser(description="Post-process and compare evaluation result JSON files.")

parser.add_argument(
    "--result_file",
    type=str,
    help="Path to the evaluation result JSON file for 'accuracy' mode.",
)
args = parser.parse_args()

def clean_trailing_dot(candidate: str) -> str:
    """29. -> 29   3.14 保持不变"""
    if candidate.endswith('.') and candidate.count('.') == 1:
        return candidate[:-1]
    return candidate

def extract_answer(sample: dict) -> str:
    # mmlu-pro answers are typically in the format "The answer is (I)."
    # We need to extract the numerical answer.
    prediction = sample["model_output"]
    extracted_ground_truth = sample['extracted_ground_truth']
    patterns = [
        r"\\boxed{(.+)}",
        r"\boxed{(.+)}"
    ]
    for pattern in patterns:
        match = re.search(pattern, prediction)
        if match:
            temp_answer = match.group(1).strip()
            is_correct = (temp_answer == extracted_ground_truth)
            if is_correct:
                return temp_answer
            elif FractionNormalizer.is_equal(temp_answer, extracted_ground_truth):
                return temp_answer

    m = re.findall(r"\$\$?(.+?)\$\$?", prediction)
    if m:
        temp_answer = m[-1].strip()
        is_correct = (temp_answer == extracted_ground_truth)
        if is_correct:
            return temp_answer
        elif FractionNormalizer.is_equal(temp_answer, extracted_ground_truth):
            return temp_answer

    patterns = [
        r"\{(.+)}",
        r"{(.+)}",
        r"\{(.+?)\}",
        r'answer is \(?([A-Z])\)?',
        r"answer is \(?([A-Z])\)?",
        r'.*[aA]nswer:\s*([A-Z])',
    ]
    # 尝试匹配每个模式
    for pattern in patterns:
        match = re.search(pattern, prediction)
        if match:
            temp_answer = match.group(1).strip()
            is_correct = (temp_answer == extracted_ground_truth)
            if is_correct:
                return temp_answer
            elif FractionNormalizer.is_equal(temp_answer, extracted_ground_truth):
                return temp_answer
            else:
                temp_answer = clean_trailing_dot(temp_answer)
                is_correct = (temp_answer == extracted_ground_truth)
                if is_correct:
                    return temp_answer

    numerical_matches = re.findall(r'[-+]?\d[\d,.]*\d*', prediction)

    # Filter out any non-numeric or malformed matches that might result from regex,
    # and ensure it starts with a digit or a sign followed by a digit.
    valid_numbers = []
    for match in numerical_matches:
        cleaned_match = match.replace(',', '')
        if re.match(r'[-+]?\d', cleaned_match):  # Simple check to ensure it's a valid number string
            valid_numbers.append(cleaned_match)

    if valid_numbers:
        for temp_answer in valid_numbers:
            temp_answer = temp_answer.strip()
            is_correct = (temp_answer == extracted_ground_truth)
            if is_correct:
                return temp_answer
            elif FractionNormalizer.is_equal(temp_answer, extracted_ground_truth):
                return temp_answer
            else:
                temp_answer = clean_trailing_dot(temp_answer)
                is_correct = (temp_answer == extracted_ground_truth)
                if is_correct:
                    return temp_answer

    final_pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(final_pattern, prediction, re.DOTALL)
    if match:
        temp_answer = match.group(0)
        is_correct = (temp_answer == extracted_ground_truth)
        if is_correct:
            return temp_answer
        elif FractionNormalizer.is_equal(temp_answer, extracted_ground_truth):
            return temp_answer
        else:
            temp_answer = clean_trailing_dot(temp_answer)
            is_correct = (temp_answer == extracted_ground_truth)
            if is_correct:
                return temp_answer
    else:
        return None

def postprocess_single(file_path=None):
    if not file_path:
        file_path = args.result_file
    try:
        print(f"Processing file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = data["records"]
        all_correct = 0
        for sample in data: # 遍历所有样本
            answers = extract_answer(sample)
            if answers:
                all_correct += 1

        # 计算并打印小数形式的准确率
        accuracy = all_correct / len(data)
        print(f"Correct: {all_correct}")
        print(f"Total: {len(data)}")
        print(f"Accuracy: {accuracy:.3f}") # 保留三位小数
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file at {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred processing {file_path}: {e}")


if not args.result_file:
    # 动态处理 0-31 层
    base_dir = "/data1/wzy/cot-mimic/results/record/hard_math_mimic_vector/"
    for i in range(0, 32):
        file_name = f"mimic_layers_{i}_direct_q_1.0.json"
        full_path = f"{base_dir}{file_name}"
        postprocess_single(full_path)
else:
    postprocess_single(args.result_file)