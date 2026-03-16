"""Evaluation metrics for benchmarks: BLEU-1, F1, LLM Judge."""

import os
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\w+', text.lower())


def calculate_bleu1(predicted: str, ground_truth: str) -> float:
    """Calculate BLEU-1 (unigram precision) score."""
    pred_tokens = tokenize(predicted)
    gt_tokens = tokenize(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    gt_counts = Counter(gt_tokens)
    pred_counts = Counter(pred_tokens)

    clipped = 0
    for token, count in pred_counts.items():
        clipped += min(count, gt_counts.get(token, 0))

    return clipped / max(len(pred_tokens), 1)


def calculate_f1(predicted: str, ground_truth: str) -> float:
    """Calculate token-level F1 score."""
    pred_tokens = set(tokenize(predicted))
    gt_tokens = set(tokenize(ground_truth))

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


def llm_judge(question: str, ground_truth: str, predicted: str, client=None) -> float:
    """Use OpenAI to judge answer quality (1-5 Likert scale).

    Args:
        client: openai.OpenAI instance
    """
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    from benchmarks.prompts import LLM_JUDGE_PROMPT

    prompt = LLM_JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        predicted=predicted,
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        score_text = response.choices[0].message.content.strip()
        score = int(re.search(r'[1-5]', score_text).group())
        return float(score)
    except Exception as e:
        print(f"LLM judge error: {e}")
        return 3.0  # default middle score on error


def binary_llm_judge(question: str, ground_truth: str, predicted: str, client=None) -> float:
    """Use OpenAI to judge answer quality on 0/1 binary scale (matching mem0's evaluation).

    Args:
        client: openai.OpenAI instance
    """
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    from benchmarks.prompts import BINARY_JUDGE_PROMPT

    prompt = BINARY_JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        predicted=predicted,
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        score_text = response.choices[0].message.content.strip()
        score = int(re.search(r'[01]', score_text).group())
        return float(score)
    except Exception as e:
        print(f"Binary judge error: {e}")
        return 0.5  # neutral default on error
