from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from statistics import mean
from typing import Any


def resolve_project_path(value: str) -> str:
    return os.path.join(os.getcwd(), value)


def read_json(path: str) -> Any:
    with open(path, "r") as handle:
        return json.load(handle)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except OSError:
        return None


def environment(runtime: str, dataset_hash: str | None = None) -> dict[str, Any]:
    return {
        "runtime": runtime,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "gitCommit": command_output(["git", "rev-parse", "HEAD"]),
        "gitDirty": bool(command_output(["git", "status", "--porcelain"]) or ""),
        "os": platform.system().lower(),
        "osRelease": platform.release(),
        "cpuModel": platform.processor() or "unknown",
        "cpuCount": os.cpu_count() or 1,
        "totalMemoryBytes": 0,
        "freeMemoryBytes": 0,
        "pythonVersion": sys.version.replace("\n", " "),
        "datasetHash": dataset_hash,
    }


def normalize_scope(scope: dict[str, Any] | None) -> dict[str, Any]:
    scope = scope or {}
    return {
      "namespace": scope.get("namespace"),
      "userId": scope.get("userId"),
      "agentId": scope.get("agentId"),
      "runId": scope.get("runId"),
    }


def scope_key(scope: dict[str, Any] | None) -> str:
    return json.dumps(normalize_scope(scope), sort_keys=True)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "stddev": 0.0,
        }
    ordered = sorted(values)
    count = len(ordered)
    avg = mean(ordered)
    variance = sum((value - avg) ** 2 for value in ordered) / count
    def quantile(ratio: float) -> float:
        index = min(count - 1, math.floor((count - 1) * ratio))
        return float(ordered[index])
    return {
        "count": count,
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": float(avg),
        "p50": quantile(0.5),
        "p95": quantile(0.95),
        "stddev": math.sqrt(variance),
    }


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def calculate_bleu1(predicted: str, ground_truth: str) -> float:
    pred_tokens = tokenize(predicted)
    gt_tokens = tokenize(ground_truth)
    if not pred_tokens or not gt_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in gt_tokens:
        counts[token] = counts.get(token, 0) + 1
    clipped = 0
    for token in pred_tokens:
        remaining = counts.get(token, 0)
        if remaining > 0:
            clipped += 1
            counts[token] = remaining - 1
    return clipped / len(pred_tokens)


def calculate_f1(predicted: str, ground_truth: str) -> float:
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


def dcg_at_k(relevances: list[int], limit: int) -> float:
    total = 0.0
    for index, relevance in enumerate(relevances[:limit]):
        total += relevance / math.log2(index + 2)
    return total


def evaluate_retrieval(ranked_ids: list[list[str]], expected_ids: list[list[str]]) -> dict[str, float]:
    total_queries = len(expected_ids)
    if total_queries == 0:
        return {
            "hitAt1": 0.0,
            "hitAt3": 0.0,
            "hitAt5": 0.0,
            "recallAt5": 0.0,
            "mrr": 0.0,
            "ndcgAt5": 0.0,
            "totalQueries": 0,
        }
    hit_1 = 0
    hit_3 = 0
    hit_5 = 0
    recall_5 = 0.0
    mrr = 0.0
    ndcg = 0.0
    for ranked, expected in zip(ranked_ids, expected_ids):
        expected_set = set(expected)
        relevances = [1 if item in expected_set else 0 for item in ranked]
        if relevances[:1] == [1]:
            hit_1 += 1
        if any(relevances[:3]):
            hit_3 += 1
        if any(relevances[:5]):
            hit_5 += 1
        hits = sum(relevances[:5])
        if expected_set:
            recall_5 += hits / len(expected_set)
        for index, value in enumerate(ranked):
            if value in expected_set:
                mrr += 1 / (index + 1)
                break
        ideal = [1] * min(5, len(expected_set))
        ideal_dcg = dcg_at_k(ideal, 5)
        ndcg += 0.0 if ideal_dcg == 0 else dcg_at_k(relevances, 5) / ideal_dcg
    return {
        "hitAt1": hit_1 / total_queries,
        "hitAt3": hit_3 / total_queries,
        "hitAt5": hit_5 / total_queries,
        "recallAt5": recall_5 / total_queries,
        "mrr": mrr / total_queries,
        "ndcgAt5": ndcg / total_queries,
        "totalQueries": total_queries,
    }


def derive_expected_ids_from_answer(turns: list[dict[str, str]], answer: str) -> list[str]:
    answer_tokens = set(tokenize(answer))
    overlaps = []
    for turn in turns:
        overlap = len([token for token in tokenize(turn["text"]) if token in answer_tokens])
        if overlap > 0:
            overlaps.append({"id": turn["id"], "overlap": overlap})
    overlaps.sort(key=lambda item: item["overlap"], reverse=True)
    best = overlaps[0]["overlap"] if overlaps else 0
    return [item["id"] for item in overlaps if item["overlap"] == best]


def approximate_disk_bytes(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0
