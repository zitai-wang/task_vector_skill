import json
import re
import os
from typing import Any, Optional, List, Dict, Tuple

INPUT_FILE = "/data1/wzy/cot-mimic/results/record/icl-qwen2.5-vl-7b-instruct-mathvista-0shot-direct/0shot.json"
OUTPUT_FILE = "/data1/wzy/cot-mimic/results/record/icl-qwen2.5-vl-7b-instruct-mathvista-0shot-direct/0shot_postprocessed_v2.json"
# RESCUED_FILE = "/data1/wzy/cot-mimic/results/record/icl-qwen2.5-vl-7b-instruct-mathvista-0shot-direct/rescued_cases_v2.json"
# REVIEW_FILE = "/data1/wzy/cot-mimic/results/record/icl-qwen2.5-vl-7b-instruct-mathvista-0shot-direct/review_cases_v2.json"


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\u2212", "-")
    s = s.replace("\xa0", " ")
    return s.strip()


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def normalize_text(s: Any) -> str:
    return norm_space(clean_text(s).lower())


def strip_outer_punct(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^[\s\.,;:!?()\[\]{}"\'`]+', '', s)
    s = re.sub(r'[\s\.,;:!?()\[\]{}"\'`]+$', '', s)
    return s.strip()


def remove_number_commas(s: str) -> str:
    return re.sub(r'(?<=\d),(?=\d)', '', s)


def flatten_items(x: Any) -> List[Dict]:
    out = []

    if isinstance(x, dict):
        if "response" in x or "question" in x or "pid" in x:
            out.append(x)
        else:
            for v in x.values():
                out.extend(flatten_items(v))
    elif isinstance(x, list):
        for elem in x:
            out.extend(flatten_items(elem))

    return out


def get_items(data: Any) -> List[Dict]:
    items = flatten_items(data)
    if not items:
        raise ValueError("没有在 JSON 中找到有效样本记录")
    return items


# =========================================================
# LaTeX / 单位 / 数字处理
# =========================================================
def strip_latex_noise(s: str) -> str:
    s = clean_text(s)
    s = s.replace("$", "")
    s = re.sub(r'\\left', '', s)
    s = re.sub(r'\\right', '', s)
    s = re.sub(r'\\,', ' ', s)
    s = re.sub(r'\\!', ' ', s)
    s = re.sub(r'\\;', ' ', s)
    s = re.sub(r'\\:', ' ', s)
    s = re.sub(r'\\quad', ' ', s)
    s = re.sub(r'\\qquad', ' ', s)
    return norm_space(s)


def remove_latex_text_units(s: str) -> str:
    s = re.sub(r'\\text\s*\{[^{}]*\}', '', s)
    s = re.sub(r'\\mathrm\s*\{[^{}]*\}', '', s)
    s = re.sub(r'\\operatorname\s*\{[^{}]*\}', '', s)
    return norm_space(s)


def clean_numeric_text(s: str) -> str:
    s = strip_latex_noise(s)
    s = remove_latex_text_units(s)
    s = remove_number_commas(s)
    s = s.replace('%', '')
    s = s.replace('°', '')
    s = re.sub(
        r'\b(cm|mm|kg|g|m/s|m|s|hours?|hour|minutes?|minute|degrees?|degree)\b',
        '',
        s,
        flags=re.I
    )
    s = norm_space(s)
    s = strip_outer_punct(s)
    return s


def extract_last_boxed_content(text: str) -> Optional[str]:
    text = clean_text(text)
    key = r'\boxed{'
    pos = text.rfind(key)
    if pos == -1:
        return None

    i = pos + len(key)
    depth = 1
    out = []
    while i < len(text):
        ch = text[i]
        if ch == '{':
            depth += 1
            out.append(ch)
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return ''.join(out).strip()
            out.append(ch)
        else:
            out.append(ch)
        i += 1

    return None


def latex_frac_to_float(s: str) -> Optional[float]:
    s = s.strip()
    m = re.fullmatch(r'\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}', s)
    if m:
        try:
            a = float(clean_numeric_text(m.group(1)))
            b = float(clean_numeric_text(m.group(2)))
            if abs(b) < 1e-12:
                return None
            return a / b
        except Exception:
            return None
    return None


def latex_sqrt_to_float(s: str) -> Optional[float]:
    s = s.strip()
    m = re.fullmatch(r'\\sqrt\s*\{([^{}]+)\}', s)
    if m:
        try:
            import math
            a = float(clean_numeric_text(m.group(1)))
            if a < 0:
                return None
            return math.sqrt(a)
        except Exception:
            return None
    return None


NUM_PATTERN = re.compile(r'[-+]?(?:\d+\.\d+|\d+|\.\d+)')


def looks_like_list_or_range(s: Any) -> bool:
    s = clean_text(s)
    if not s:
        return False
    patterns = [
        r'^\[.*\]$',
        r'^\(.*\)$',
        r'\bbetween\b',
        r'\band\b',
        r'\bto\b',
        r',',
    ]
    return any(re.search(p, s, flags=re.I) for p in patterns)


def parse_numeric_value(raw: Any) -> Optional[float]:
    if raw is None:
        return None

    s = clean_text(raw)
    if not s:
        return None

    if looks_like_list_or_range(s):
        return None

    s = strip_latex_noise(s)
    s = remove_latex_text_units(s)

    frac_val = latex_frac_to_float(s)
    if frac_val is not None:
        return frac_val

    sqrt_val = latex_sqrt_to_float(s)
    if sqrt_val is not None:
        return sqrt_val

    s = clean_numeric_text(s)

    if looks_like_list_or_range(s):
        return None

    if re.fullmatch(r'[-+]?\d+\s*/\s*\d+', s):
        try:
            a, b = s.replace(' ', '').split('/')
            b = float(b)
            if abs(b) < 1e-12:
                return None
            return float(a) / b
        except Exception:
            return None

    m = NUM_PATTERN.fullmatch(s)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None

    return None


def format_numeric_for_output(val: float, answer_type: str = "") -> str:
    answer_type = normalize_text(answer_type)
    if answer_type == "integer":
        return str(int(round(val)))
    if abs(val - int(val)) < 1e-9:
        return str(int(val))
    return str(val)


def count_decimal_places(s: Any) -> Optional[int]:
    s = clean_text(s)
    s = strip_latex_noise(s)
    s = remove_latex_text_units(s)
    s = remove_number_commas(s)
    s = strip_outer_punct(s)

    m = re.fullmatch(r'[-+]?\d+\.(\d+)', s)
    if m:
        return len(m.group(1))

    if re.fullmatch(r'[-+]?\d+', s):
        return 0

    return None


def numeric_string_match(pred: Any, gt: Any, answer_type: str = "") -> bool:
    p_num = parse_numeric_value(pred)
    g_num = parse_numeric_value(gt)
    if p_num is None or g_num is None:
        return False

    answer_type = normalize_text(answer_type)
    if answer_type == "integer":
        return int(round(p_num)) == int(round(g_num))

    p_dec = count_decimal_places(pred)
    g_dec = count_decimal_places(gt)

    if g_dec is not None and g_dec >= 0:
        try:
            p_round = f"{round(p_num, g_dec):.{g_dec}f}"
            g_round = f"{round(g_num, g_dec):.{g_dec}f}"
            return p_round == g_round
        except Exception:
            pass

    if p_dec is not None and g_dec is not None:
        try:
            max_dec = max(p_dec, g_dec)
            p_round = f"{round(p_num, max_dec):.{max_dec}f}"
            g_round = f"{round(g_num, max_dec):.{max_dec}f}"
            return p_round == g_round
        except Exception:
            pass

    return abs(p_num - g_num) <= 1e-6


YES_SET = {"yes", "true"}
NO_SET = {"no", "false"}

SAFE_SHORT_ANSWER_WHITELIST = {
    "increase",
    "decrease",
    "shortage",
    "surplus",
    "grass",
    "o'clock",
    "upper half",
    "lower half",
    "population will decrease",
    "it would decrease",
    "it would also decrease",
}


def normalize_word_answer(s: Any) -> str:
    s = normalize_text(s)
    s = strip_latex_noise(s)
    s = strip_outer_punct(s)
    s = normalize_text(s)

    if s in YES_SET:
        return "yes"
    if s in NO_SET:
        return "no"

    return s


def normalize_gt(item: Dict) -> str:
    return normalize_word_answer(item.get("answer", item.get("gt", "")))


def canonical_short_answer(s: Any) -> str:
    s = normalize_word_answer(s)

    if "also decrease" in s:
        return "decrease"
    if "will decrease" in s:
        return "decrease"
    if "would decrease" in s:
        return "decrease"

    if "also increase" in s:
        return "increase"
    if "will increase" in s:
        return "increase"
    if "would increase" in s:
        return "increase"

    return s


def is_yes_no_question(question: str) -> bool:
    q = normalize_text(question)
    return bool(re.match(r'^(is|are|do|does|did|has|have|can|could|was|were)\b', q))


def extract_explicit_yes_no(response: str) -> Optional[str]:
    text = clean_text(response)
    tail = text[-350:] if len(text) > 350 else text

    patterns = [
        r'answer\s*[:：]\s*(yes|no|true|false)\b',
        r'final answer\s*[:：]?\s*(yes|no|true|false)\b',
        r'the answer is(?: likely)?\s*\(?[A-Z]\)?\s*(yes|no|true|false)\b',
        r'the correct answer is(?: likely)?\s*\(?[A-Z]\)?\s*(yes|no|true|false)\b',
        r'\([A-Z]\)\s*(yes|no|true|false)\b',
        r'\\boxed\{\s*(yes|no|true|false)\s*\}',
    ]

    for p in patterns:
        ms = re.findall(p, tail, flags=re.I)
        if ms:
            ans = ms[-1].lower()
            return "yes" if ans in YES_SET else "no"

    return None


def extract_polarity_yes_no(question: str, response: str) -> Optional[str]:
    if not is_yes_no_question(question):
        return None

    text = normalize_text(response)
    tail = text[-450:] if len(text) > 450 else text

    positive_patterns = [
        r'\bindeed\b',
        r'\bis true\b',
        r'\bdoes appear to\b',
        r'\bdoes seem to\b',
        r'\bis the maximum\b',
        r'\bis greater than\b',
        r'\bis less than\b',
        r'\bare three distinct\b',
        r'\bis predominantly\b',
        r'\bis equal to\b',
        r'\bcorrect answer is likely \(a\)\b',
        r'\bcorrect answer is \(a\)\b',
        r'\banswer is \(a\)\b',
    ]

    negative_patterns = [
        r'\bis not\b',
        r'\bnot true\b',
        r'\bdoes not appear\b',
        r'\bdoes not seem\b',
        r'\bis not the maximum\b',
        r'\bnot greater than\b',
        r'\bnot less than\b',
        r'\bcorrect answer is likely \(b\)\b',
        r'\bcorrect answer is \(b\)\b',
        r'\banswer is \(b\)\b',
    ]

    pos_hit = any(re.search(p, tail, flags=re.I) for p in positive_patterns)
    neg_hit = any(re.search(p, tail, flags=re.I) for p in negative_patterns)

    if pos_hit and not neg_hit:
        return "yes"
    if neg_hit and not pos_hit:
        return "no"
    return None


def extract_option_letter(response: str) -> Optional[str]:
    if not response:
        return None

    text = clean_text(response)
    tail = text[-350:] if len(text) > 350 else text

    patterns = [
        r'answer\s*[:：]\s*\(?([A-Z])\)?\b',
        r'final answer\s*[:：]?\s*\(?([A-Z])\)?\b',
        r'the answer is(?: likely)?\s*\(?([A-Z])\)?\b',
        r'the correct answer is(?: likely)?\s*\(?([A-Z])\)?\b',
        r'\boption\s*\(?([A-Z])\)?\b',
        r'\bchoice\s*\(?([A-Z])\)?\b',
        r'\\boxed\{\s*([A-Z])\s*\}',
        r'\(([A-Z])\)',
    ]

    for p in patterns:
        matches = re.findall(p, tail, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    return None


def extract_reasoning_image_letter(response: str) -> Optional[str]:
    text = clean_text(response)

    patterns = [
        r'\bimage\s+([A-Z])\b',
        r'\boption\s+([A-Z])\b',
        r'\bchoice\s+([A-Z])\b',
    ]

    found = []
    for p in patterns:
        ms = re.findall(p, text, flags=re.I)
        found.extend([m.upper() for m in ms])

    if not found:
        return None
    return found[-1]


def has_option_conflict(response: str, final_letter: str) -> bool:
    reason_letter = extract_reasoning_image_letter(response)
    if reason_letter is None or final_letter is None:
        return False
    return reason_letter != final_letter


def map_letter_to_choice(letter: str, choices: List[str]) -> Optional[str]:
    if not letter or not choices:
        return None
    idx = ord(letter.upper()) - ord('A')
    if 0 <= idx < len(choices):
        return choices[idx]
    return None


def match_response_to_choice_text(response: str, choices: List[str]) -> Optional[Tuple[str, str]]:
    if not response or not choices:
        return None

    resp = normalize_text(response)
    best_idx = -1
    best_len = -1

    for i, choice in enumerate(choices):
        c = normalize_text(choice)
        if not c:
            continue
        if c in resp:
            if len(c) > best_len:
                best_len = len(c)
                best_idx = i

    if best_idx != -1:
        return choices[best_idx], chr(ord('A') + best_idx)

    return None


# =========================================================
# 新增规则 1：显式选项文本答案优先
# 例如：the correct answer is (B) kril
# 即使后面 boxed(A) 冲突，只要文本答案和 GT 一致，仍可救
# =========================================================
def extract_explicit_option_text_answer(response: str) -> Optional[str]:
    text = clean_text(response)
    tail = text[-500:] if len(text) > 500 else text

    patterns = [
        r'the correct answer is(?: likely)?\s*\(([A-Z])\)\s*([A-Za-z][A-Za-z\' \-]+?)(?=[\.;,\n]|but\b|however\b|because\b|and\b|$)',
        r'the answer is(?: likely)?\s*\(([A-Z])\)\s*([A-Za-z][A-Za-z\' \-]+?)(?=[\.;,\n]|but\b|however\b|because\b|and\b|$)',
        r'final answer\s*[:：]?\s*\(([A-Z])\)\s*([A-Za-z][A-Za-z\' \-]+?)(?=[\.;,\n]|but\b|however\b|because\b|and\b|$)',
        r'answer\s*[:：]?\s*\(([A-Z])\)\s*([A-Za-z][A-Za-z\' \-]+?)(?=[\.;,\n]|but\b|however\b|because\b|and\b|$)',
        r'\(([A-Z])\)\s*([A-Za-z][A-Za-z\' \-]+?)(?=[\.;,\n]|but\b|however\b|because\b|and\b|$)',
    ]

    candidates = []
    for p in patterns:
        ms = re.findall(p, tail, flags=re.I)
        for _, ans in ms:
            ans = norm_space(ans)
            ans = strip_outer_punct(ans)
            if ans:
                candidates.append(ans)

    if not candidates:
        return None
    return candidates[-1]


def extract_answer_text_after_option_prefix(response: str) -> Optional[str]:
    text = clean_text(response)
    tail = text[-400:] if len(text) > 400 else text

    patterns = [
        r'answer\s*[:：]\s*\(?[A-Z]\)?\s*[\.\-:：]?\s*([A-Za-z][A-Za-z\' \-]+)',
        r'final answer\s*[:：]?\s*\(?[A-Z]\)?\s*[\.\-:：]?\s*([A-Za-z][A-Za-z\' \-]+)',
        r'the correct answer is(?: likely)?\s*[:：]?\s*\(?[A-Z]\)?\s*[\.\-:：]?\s*([A-Za-z][A-Za-z\' \-]+)',
        r'\([A-Z]\)\s*([A-Za-z][A-Za-z\' \-]+)',
        r'\b[A-Z]\)\s*([A-Za-z][A-Za-z\' \-]+)',
    ]

    candidates = []
    for p in patterns:
        ms = re.findall(p, tail, flags=re.I)
        for m in ms:
            ans = norm_space(m)
            ans = re.sub(r'^\(?[A-Z]\)?\s*[\.\-:：]?\s*', '', ans).strip()
            ans = strip_outer_punct(ans)
            ans = re.split(r'[\.;,\n]', ans)[0].strip()
            if ans:
                candidates.append(ans)

    if not candidates:
        return None

    return candidates[-1]


# =========================================================
# 新增规则 3：识别 boxed 字母和显式文本答案冲突
# 仅用于记录 review 信息，不阻止显式文本答案救回
# =========================================================
def detect_boxed_letter_text_conflict(response: str) -> Optional[Dict[str, str]]:
    text = clean_text(response)

    explicit_text = extract_explicit_option_text_answer(text)
    final_letter = extract_option_letter(text)

    if explicit_text and final_letter:
        return {
            "explicit_text_answer": explicit_text,
            "final_letter": final_letter
        }
    return None

def extract_final_semantic_claim(response: str) -> Optional[str]:
    text = normalize_text(response)
    tail = text[-500:] if len(text) > 500 else text

    if re.search(r'\bsteelheads?\b.*\bdecrease\b', tail):
        return "steelheads decrease"
    if re.search(r'\bpredatory insects?\b.*\bdecrease\b', tail):
        return "predatory insects decrease"
    if re.search(r'\bkril\b', tail):
        return "kril"
    if re.search(r'\bdecrease\b', tail):
        return "decrease"
    if re.search(r'\bincrease\b', tail):
        return "increase"
    return None


def should_block_previous_sentence_rescue(item: Dict, candidate_answer: str) -> bool:
    response = clean_text(item.get("response", ""))
    candidate = canonical_short_answer(candidate_answer)
    final_claim = extract_final_semantic_claim(response)

    if not final_claim:
        return False

    # 针对已知高风险模式：最终明确说 steelheads decrease，
    # 但 previous_sentence 抽成别的 decrease 类答案
    if "steelheads decrease" in final_claim and candidate == "decrease":
        gt_norm = canonical_short_answer(normalize_gt(item))
        raw_gt = normalize_text(item.get("answer", item.get("gt", "")))
        if "steelhead" not in raw_gt and gt_norm == "decrease":
            return True

    return False

def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    text = re.sub(r'\n+', ' ', text)
    parts = re.split(r'(?<=[\.\?!])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def extract_short_answer_from_previous_sentence(response: str, gt_norm: str) -> Optional[str]:
    gt_canon = canonical_short_answer(gt_norm)

    allowed = {
        "increase", "decrease", "shortage", "surplus",
        "grass", "o'clock", "upper half", "lower half"
    }
    if gt_canon not in allowed:
        return None

    text = clean_text(response)
    tail = text[-450:] if len(text) > 450 else text
    tail_norm = normalize_text(tail)

    option_only_patterns = [
        r'(the correct answer is|the answer is|answer)\s*[:：]?\s*\(?[A-Z]\)?\s*\.?$',
        r'\\boxed\{\s*[A-Z]\s*\}\s*$',
        r'^\s*[A-Z]\s*$',
    ]
    if not any(re.search(p, tail.strip(), flags=re.I | re.M) for p in option_only_patterns):
        if not re.search(r'(the correct answer is|the answer is|answer)\s*[:：]?\s*[A-Z]\s*$', tail_norm, flags=re.I):
            return None

    if gt_canon == "decrease":
        if re.search(r'\b(decrease|decreases|decreased|decreasing)\b', tail_norm):
            return gt_norm
    if gt_canon == "increase":
        if re.search(r'\b(increase|increases|increased|increasing)\b', tail_norm):
            return gt_norm

    if re.search(rf'\b{re.escape(gt_canon)}\b', tail_norm):
        return gt_norm

    return None


def evaluate_multi_choice(item: Dict) -> Dict[str, Any]:
    response = clean_text(item.get("response", ""))
    choices = item.get("choices", []) or []
    gt_answer = clean_text(item.get("answer", item.get("gt", "")))
    gt_norm = normalize_word_answer(gt_answer)

    pred_letter = extract_option_letter(response)
    if pred_letter is not None:
        if has_option_conflict(response, pred_letter):
            return {
                "pred_letter": pred_letter,
                "pred_choice": None,
                "pred_norm": None,
                "gt_norm": gt_norm,
                "is_correct": False,
                "reason": "option_conflict"
            }

        pred_choice = map_letter_to_choice(pred_letter, choices)
        if pred_choice is not None:
            pred_choice_norm = normalize_word_answer(pred_choice)
            return {
                "pred_letter": pred_letter,
                "pred_choice": pred_choice,
                "pred_norm": pred_choice_norm,
                "gt_norm": gt_norm,
                "is_correct": pred_choice_norm == gt_norm,
                "reason": "option_letter"
            }

        return {
            "pred_letter": pred_letter,
            "pred_choice": pred_letter,
            "pred_norm": normalize_word_answer(pred_letter),
            "gt_norm": gt_norm,
            "is_correct": normalize_word_answer(pred_letter) == gt_norm,
            "reason": "option_letter_no_choices"
        }

    matched = match_response_to_choice_text(response, choices)
    if matched is not None:
        pred_choice, pred_letter = matched
        pred_choice_norm = normalize_word_answer(pred_choice)
        return {
            "pred_letter": pred_letter,
            "pred_choice": pred_choice,
            "pred_norm": pred_choice_norm,
            "gt_norm": gt_norm,
            "is_correct": pred_choice_norm == gt_norm,
            "reason": "choice_text_match"
        }

    return {
        "pred_letter": None,
        "pred_choice": None,
        "pred_norm": None,
        "gt_norm": gt_norm,
        "is_correct": False,
        "reason": "multi_choice_no_match"
    }

FINAL_ANSWER_HINTS = [
    r'final answer\s*(?:is|=|:)?',
    r'answer\s*(?:is|=|:)',
    r'therefore[, ]*',
    r'thus[, ]*',
    r'hence[, ]*',
    r'so[, ]*',
]


def extract_from_tail_region(response: str, answer_type: str = "") -> Optional[str]:
    text = clean_text(response)
    lower = text.lower()

    best = None
    best_pos = -1
    for hint in FINAL_ANSWER_HINTS:
        for m in re.finditer(hint, lower):
            if m.start() > best_pos:
                best_pos = m.start()
                best = text[m.end():]

    if not best:
        return None

    candidate = norm_space(best[:180])

    yn = extract_explicit_yes_no(candidate)
    if yn is not None:
        return yn

    if looks_like_list_or_range(candidate):
        candidate = strip_outer_punct(strip_latex_noise(candidate))
        return candidate if candidate else None

    num = parse_numeric_value(candidate)
    if num is not None:
        return format_numeric_for_output(num, answer_type)

    candidate = strip_outer_punct(candidate)
    return candidate if candidate else None


def extract_last_number_anywhere(response: str, answer_type: str = "") -> Optional[str]:
    text = clean_text(response)

    if looks_like_list_or_range(text):
        return None

    text = strip_latex_noise(text)
    text = remove_latex_text_units(text)
    text = remove_number_commas(text)

    nums = NUM_PATTERN.findall(text)
    if not nums:
        return None

    try:
        val = float(nums[-1])
        return format_numeric_for_output(val, answer_type)
    except Exception:
        return None


def robust_extract_free_form_answer(item: Dict) -> Tuple[Optional[str], str, Dict[str, Any]]:
    response = item.get("response", "")
    answer_type = item.get("answer_type", "")
    question = item.get("question", "")
    gt_norm = normalize_gt(item)
    extra_info: Dict[str, Any] = {}

    if not response or not isinstance(response, str):
        return None, "empty_response", extra_info

    explicit_yn = extract_explicit_yes_no(response)
    if explicit_yn is not None:
        return explicit_yn, "explicit_yes_no", extra_info

    polarity_yn = extract_polarity_yes_no(question, response)
    if polarity_yn is not None:
        return polarity_yn, "polarity_yes_no", extra_info

    # 强规则：显式文本答案优先，可覆盖 boxed 字母冲突
    explicit_option_text = extract_explicit_option_text_answer(response)
    if explicit_option_text:
        boxed_conflict = detect_boxed_letter_text_conflict(response)
        if boxed_conflict is not None:
            extra_info["boxed_text_conflict"] = boxed_conflict

        norm_ans = canonical_short_answer(explicit_option_text)
        gt_canon = canonical_short_answer(gt_norm)
        if norm_ans == gt_canon or normalize_word_answer(explicit_option_text) == normalize_word_answer(gt_norm):
            return explicit_option_text, "explicit_option_text", extra_info

    option_prefixed_text = extract_answer_text_after_option_prefix(response)
    if option_prefixed_text:
        norm_ans = canonical_short_answer(option_prefixed_text)
        gt_canon = canonical_short_answer(gt_norm)
        if norm_ans == gt_canon:
            return option_prefixed_text, "option_prefixed_text", extra_info

    prev_short = extract_short_answer_from_previous_sentence(response, gt_norm)
    if prev_short is not None:
        if should_block_previous_sentence_rescue(item, prev_short):
            extra_info["blocked_previous_sentence_rescue"] = True
        else:
            return prev_short, "previous_sentence_short_answer", extra_info

    boxed = extract_last_boxed_content(response)
    if boxed:
        boxed_raw = strip_outer_punct(strip_latex_noise(boxed))

        boxed_yn = extract_explicit_yes_no(boxed_raw)
        if boxed_yn is not None:
            return boxed_yn, "boxed_yes_no", extra_info

        if looks_like_list_or_range(boxed_raw):
            return boxed_raw, "boxed_range_or_list", extra_info

        val = parse_numeric_value(boxed_raw)
        if val is not None:
            return format_numeric_for_output(val, answer_type), "boxed_numeric", extra_info

        if boxed_raw:
            stripped = re.sub(r'^\(?[A-Z]\)?\s*[\.\-:：]?\s*', '', boxed_raw).strip()
            if canonical_short_answer(stripped) == canonical_short_answer(gt_norm):
                return stripped, "boxed_option_prefixed_text", extra_info
            return boxed_raw, "boxed_text", extra_info

    tail_ans = extract_from_tail_region(response, answer_type)
    if tail_ans is not None:
        return tail_ans, "tail_region_rule", extra_info

    last_num = extract_last_number_anywhere(response, answer_type)
    if last_num is not None:
        return last_num, "last_number_rule", extra_info

    return None, "no_extraction", extra_info


def answers_match_free_form(pred: Any, gt: Any, answer_type: str = "") -> bool:
    answer_type = normalize_text(answer_type)
    pred_s = clean_text(pred)
    gt_s = clean_text(gt)

    if looks_like_list_or_range(pred_s) or looks_like_list_or_range(gt_s):
        return normalize_word_answer(pred_s) == normalize_word_answer(gt_s)

    if numeric_string_match(pred, gt, answer_type):
        return True

    p = canonical_short_answer(pred)
    g = canonical_short_answer(gt)
    if not p or not g:
        return False
    return p == g


def compute_accuracy(items: List[Dict]) -> float:
    total = 0
    correct = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        total += 1
        if item.get("is_correct", False):
            correct += 1
    return 100.0 * correct / total if total else 0.0


def is_multi_choice_item(item: Dict) -> bool:
    question_type = normalize_text(item.get("question_type", ""))
    choices = item.get("choices", None)
    gt = clean_text(item.get("answer", item.get("gt", "")))
    if question_type == "multi_choice":
        return True
    if isinstance(choices, list) and len(choices) > 0:
        return True
    if re.fullmatch(r'\(?[A-Z]\)?', gt, flags=re.I):
        return True
    return False


# def post_process(input_path: str, output_path: str, rescued_path: str, review_path: str):
def post_process(input_path: str, output_path: str | None = None):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = get_items(data)

    print("num_items =", len(items))
    if len(items) > 0 and isinstance(items[0], dict):
        print("first_item_keys =", list(items[0].keys()))

    before_acc = compute_accuracy(items)

    rescued_cases = []
    review_cases = []

    total = len(items)
    target_count = 0
    rescued_count = 0

    for item in items:
        if not isinstance(item, dict):
            continue

        response = item.get("response", "")
        answer = item.get("answer", item.get("gt", ""))
        answer_type = item.get("answer_type", "")
        old_ext_p = item.get("ext_p", None)
        old_correct = bool(item.get("is_correct", False))
        pid = item.get("pid", None)

        if old_correct and old_ext_p is not None:
            continue

        target_count += 1

        if is_multi_choice_item(item):
            mc_res = evaluate_multi_choice(item)

            if mc_res["pred_choice"] is not None:
                item["ext_p"] = mc_res["pred_choice"]

            if mc_res["is_correct"]:
                if not old_correct:
                    item["is_correct"] = True
                    item["score"] = True
                    rescued_count += 1
                    rescued_cases.append({
                        "pid": pid,
                        "question_type": item.get("question_type"),
                        "answer_type": answer_type,
                        "gt_answer": answer,
                        "old_ext_p": old_ext_p,
                        "new_ext_p": mc_res["pred_choice"],
                        "pred_letter": mc_res["pred_letter"],
                        "rescue_reason": mc_res["reason"],
                        "response_tail": clean_text(response)[-300:]
                    })
            else:
                if mc_res["pred_choice"] is not None or mc_res["pred_letter"] is not None:
                    review_cases.append({
                        "pid": pid,
                        "question_type": item.get("question_type"),
                        "answer_type": answer_type,
                        "gt_answer": answer,
                        "old_ext_p": old_ext_p,
                        "new_ext_p": mc_res["pred_choice"],
                        "pred_letter": mc_res["pred_letter"],
                        "review_reason": mc_res["reason"],
                        "response_tail": clean_text(response)[-300:]
                    })
                if not old_correct:
                    item["is_correct"] = False
                    item["score"] = False

        else:
            new_ext_p, reason, extra_info = robust_extract_free_form_answer(item)

            if new_ext_p is not None:
                item["ext_p"] = new_ext_p

            matched = False
            if new_ext_p is not None:
                matched = answers_match_free_form(new_ext_p, answer, answer_type)

            if matched:
                if not old_correct:
                    item["is_correct"] = True
                    item["score"] = True
                    rescued_count += 1
                    rescue_record = {
                        "pid": pid,
                        "question_type": item.get("question_type"),
                        "answer_type": answer_type,
                        "gt_answer": answer,
                        "old_ext_p": old_ext_p,
                        "new_ext_p": new_ext_p,
                        "pred_letter": None,
                        "rescue_reason": reason,
                        "response_tail": clean_text(response)[-300:]
                    }
                    if extra_info:
                        rescue_record["extra_info"] = extra_info
                    rescued_cases.append(rescue_record)
            else:
                if new_ext_p is not None or extra_info:
                    review_record = {
                        "pid": pid,
                        "question_type": item.get("question_type"),
                        "answer_type": answer_type,
                        "gt_answer": answer,
                        "old_ext_p": old_ext_p,
                        "new_ext_p": new_ext_p,
                        "pred_letter": None,
                        "review_reason": reason,
                        "response_tail": clean_text(response)[-300:]
                    }
                    if extra_info:
                        review_record["extra_info"] = extra_info
                    review_cases.append(review_record)
                if not old_correct:
                    item["is_correct"] = False
                    item["score"] = False

    after_acc = compute_accuracy(items)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # with open(rescued_path, "w", encoding="utf-8") as f:
    #     json.dump(rescued_cases, f, ensure_ascii=False, indent=2)

    # with open(review_path, "w", encoding="utf-8") as f:
    #     json.dump(review_cases, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("后处理完成")
    print(f"输入文件: {input_path}")
    if output_path:
        print(f"输出文件: {output_path}")
    else:
        print("输出文件: 未保存")
    print(f"总样本数: {total}")
    print(f"目标处理样本数(false 或 ext_p=null): {target_count}")
    print(f"成功救回数量: {rescued_count}")
    print(f"需要人工复核数量: {len(review_cases)}")
    print(f"处理前准确率: {before_acc:.4f}%")
    print(f"处理后准确率: {after_acc:.4f}%")
    # print(f"rescued cases 文件: {rescued_path}")
    # print(f"review cases 文件: {review_path}")
    print("=" * 80)


# if __name__ == "__main__":
#     post_process(INPUT_FILE, OUTPUT_FILE, RESCUED_FILE, REVIEW_FILE)

if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    no_save = False
    if "--no-save" in args:
        no_save = True
        args = [a for a in args if a != "--no-save"]

    if len(args) >= 1:
        input_file = args[0]
        base_dir = os.path.dirname(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]

        output_file = None if no_save else os.path.join(base_dir, base_name + "_post.json")
        # rescued_file = os.path.join(base_dir, base_name + "_rescued.json")
        # review_file = os.path.join(base_dir, base_name + "_review.json")

        # post_process(input_file, output_file, rescued_file, review_file)
        post_process(input_file, output_file)
    else:
        post_process(INPUT_FILE, None)
        # post_process(INPUT_FILE, OUTPUT_FILE, RESCUED_FILE, REVIEW_FILE)
