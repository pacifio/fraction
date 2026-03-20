from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.getcwd())

from benchmarks.prompts import ANSWER_PROMPT, BINARY_JUDGE_PROMPT, LLM_JUDGE_PROMPT
from benchmarks.parity.normalize import (
    approximate_disk_bytes,
    calculate_bleu1,
    calculate_f1,
    command_output,
    derive_expected_ids_from_answer,
    environment,
    evaluate_retrieval,
    file_sha256,
    normalize_scope,
    read_json,
    resolve_project_path,
    scope_key,
    summarize,
)


REQUIRED_MODULES = ["pydantic", "spacy", "sentence_transformers", "usearch", "llmlingua"]


def required_modules_for_suite(suite: dict[str, Any]) -> list[str]:
    required = {"pydantic", "spacy", "sentence_transformers", "usearch", "rank_bm25"}
    for scenario in suite["scenarios"]:
        common = scenario.get("config", {}).get("common", {})
        runtime = scenario.get("config", {}).get("python", {})
        compressor_type = str(runtime.get("compressorType", common.get("compressorType", "llmlingua2")))
        if compressor_type in {"llmlingua2", "ensemble"}:
            required.add("llmlingua")
        if compressor_type == "llm":
            required.add("litellm")
    return sorted(required)


def dependency_error(suite: dict[str, Any]) -> str | None:
    missing: list[str] = []
    for module in required_modules_for_suite(suite):
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    try:
        import spacy

        spacy.load("en_core_web_sm")
    except Exception:
        missing.append("en_core_web_sm")
    if not missing:
        return None
    return f"Missing Python benchmark dependencies: {', '.join(missing)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Fraction parity benchmarks for Python")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_fraction_modules():
    from fraction import FractionConfig, Memory
    return FractionConfig, Memory


class IdentityCompressor:
    def score(self, text: str) -> list[tuple[str, float]]:
        return [(token, 1.0) for token in text.split()]

    def compress(self, text: str, rate: float = 1.0, adaptive: bool = False):
        from fraction.types import CompressedFragment

        return CompressedFragment(
            original_text=text,
            compressed_text=text,
            compression_ratio=1.0,
            token_scores=self.score(text),
            entities=[],
        )


def create_memory(scenario: dict[str, Any], filename_root: str):
    FractionConfig, Memory = load_fraction_modules()
    config_kwargs = dict(scenario.get("config", {}).get("common", {}))
    config_kwargs.update(scenario.get("config", {}).get("python", {}))
    compressor_type = str(config_kwargs.get("compressorType", "llmlingua2"))
    if compressor_type == "off":
        import fraction.memory as fraction_memory

        fraction_memory.build_compressor = lambda config: IdentityCompressor()
    config = FractionConfig(
        compression_rate=float(config_kwargs.get("compressionRate", 0.6)),
        compressor_type="llmlingua2" if compressor_type == "off" else compressor_type,
        top_k=int(config_kwargs.get("topK", 5)),
        vector_store_path=f"{filename_root}.usearch",
        metadata_path=f"{filename_root}_metadata.json",
        history_db_path=f"{filename_root}_history.db",
    )
    return Memory(data_dir=str(Path(filename_root).parent / "python-data"), config=config, auto_save=False)


def cleanup_python_memory(filename_root: str) -> None:
    for suffix in [".usearch", "_metadata.json", "_history.db", "_graph.json"]:
        path = f"{filename_root}{suffix}"
        try:
            os.remove(path)
        except OSError:
            pass
    data_dir = Path(filename_root).parent / "python-data"
    if data_dir.exists():
        for child in data_dir.iterdir():
            try:
                child.unlink()
            except OSError:
                pass
        try:
            data_dir.rmdir()
        except OSError:
            pass


def to_python_scope(scope: dict[str, Any] | None) -> dict[str, Any]:
    normalized = normalize_scope(scope)
    return {
        "user_id": normalized.get("userId"),
        "agent_id": normalized.get("agentId"),
        "run_id": normalized.get("runId"),
    }


def search_ids(results: dict[str, Any]) -> list[str]:
    return [str(entry.get("id", "")) for entry in results.get("results", [])]


def map_expected_ids(ids: list[str], fixture_to_runtime_ids: dict[str, str]) -> list[str]:
    return [fixture_to_runtime_ids.get(memory_id, memory_id) for memory_id in ids]


def group_by_scope(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = scope_key(record.get("scope"))
        grouped.setdefault(key, []).append(record)
    return grouped


def run_synthetic_scenario(suite: dict[str, Any], scenario: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    dataset = read_json(resolve_project_path(scenario["workload"]["dataset"]["path"]))
    warmup = int(scenario.get("warmupRuns", suite["warmupRuns"]))
    measured = int(scenario.get("measuredRuns", suite["measuredRuns"]))
    timings = {key: [] for key in ["openMs", "addMs", "addManyMs", "updateMs", "searchMs", "getMs", "getAllMs", "deleteMs", "deleteAllMs"]}
    ranked_ids: list[list[str]] = []
    expected_ids: list[list[str]] = []
    notes: list[str] = []
    parity_warnings = []
    config_common = scenario.get("config", {}).get("common", {})
    config_runtime = scenario.get("config", {}).get("python", {})
    compressor_type = str(config_runtime.get("compressorType", config_common.get("compressorType", "llmlingua2")))
    if scenario["mode"] == "best-effort-local" and compressor_type != "off":
        parity_warnings.append("Python strict suite uses runtime-native local compression configuration.")

    def run_once(index: int, record_metrics: bool) -> int:
        filename_root = str(Path(".benchmark-parity") / "py" / f"{suite['suiteId']}-{scenario['id']}" / f"run-{index}")
        Path(filename_root).parent.mkdir(parents=True, exist_ok=True)
        cleanup_python_memory(filename_root)
        open_start = time.perf_counter()
        memory = create_memory(scenario, filename_root)
        if record_metrics:
            timings["openMs"].append((time.perf_counter() - open_start) * 1000)
        try:
            fixture_to_runtime_ids: dict[str, str] = {}
            for record in dataset["records"]:
                started = time.perf_counter()
                scope = to_python_scope(record.get("scope"))
                created = memory.add(record["content"], metadata=record.get("metadata"), **scope)
                if record_metrics:
                    timings["addMs"].append((time.perf_counter() - started) * 1000)
                created_rows = created.get("results", [])
                if created_rows:
                    fixture_to_runtime_ids[record["id"]] = str(created_rows[0].get("id", record["id"]))
            for query in dataset["queries"]:
                started = time.perf_counter()
                scope = to_python_scope(query.get("scope"))
                results = memory.search(
                    query["query"],
                    limit=int(scenario.get("topK", suite["topK"])),
                    filters={query["filter"]["field"]: query["filter"]["value"]} if query.get("filter") else None,
                    **scope,
                )
                if record_metrics:
                    timings["searchMs"].append((time.perf_counter() - started) * 1000)
                    ranked_ids.append(search_ids(results))
                    expected_ids.append(map_expected_ids(list(query["expectedIds"]), fixture_to_runtime_ids))
            for record in dataset["records"][:3]:
                started = time.perf_counter()
                memory.get(fixture_to_runtime_ids.get(record["id"], record["id"]))
                if record_metrics:
                    timings["getMs"].append((time.perf_counter() - started) * 1000)
            if scenario["workload"].get("includeGetAll"):
                for scoped_records in group_by_scope(dataset["records"]).values():
                    scope = to_python_scope(scoped_records[0].get("scope"))
                    started = time.perf_counter()
                    memory.get_all(**scope)
                    if record_metrics:
                        timings["getAllMs"].append((time.perf_counter() - started) * 1000)
            if scenario["workload"].get("includeUpdate"):
                for mutation in dataset.get("mutations", {}).get("updates", []):
                    started = time.perf_counter()
                    updated = memory.update(
                        fixture_to_runtime_ids.get(mutation["id"], mutation["id"]),
                        mutation["content"],
                    )
                    if record_metrics:
                        timings["updateMs"].append((time.perf_counter() - started) * 1000)
                    if mutation.get("expectedContentIncludes") and mutation["expectedContentIncludes"] not in updated.get("memory", ""):
                        notes.append(f"Update expectation failed for {mutation['id']}")
            for mutation in dataset.get("mutations", {}).get("deletes", []):
                started = time.perf_counter()
                memory.delete(fixture_to_runtime_ids.get(mutation["id"], mutation["id"]))
                if record_metrics:
                    timings["deleteMs"].append((time.perf_counter() - started) * 1000)
            if scenario["workload"].get("includeDeleteAll"):
                for mutation in dataset.get("mutations", {}).get("deleteAll", []):
                    started = time.perf_counter()
                    memory.delete_all(**to_python_scope(mutation.get("scope")))
                    if record_metrics:
                        timings["deleteAllMs"].append((time.perf_counter() - started) * 1000)
            memory.save()
            return (
                approximate_disk_bytes(f"{filename_root}.usearch")
                + approximate_disk_bytes(f"{filename_root}_metadata.json")
                + approximate_disk_bytes(f"{filename_root}_history.db")
            )
        finally:
            cleanup_python_memory(filename_root)

    for warm in range(warmup):
        run_once(-1 - warm, False)

    disk_bytes = 0
    for run_index in range(measured):
        disk_bytes = max(disk_bytes, run_once(run_index, True))

    return {
        "scenarioId": scenario["id"],
        "runtime": "python",
        "tier": suite["tier"],
        "mode": scenario["mode"],
        "status": "passed",
        "config": {**scenario.get("config", {}).get("common", {}), **scenario.get("config", {}).get("python", {})},
        "environment": env,
        "timings": {key: summarize(values) for key, values in timings.items()},
        "throughput": {
            "addOpsPerSec": len(timings["addMs"]) / (sum(timings["addMs"]) / 1000) if timings["addMs"] else None,
            "searchOpsPerSec": len(timings["searchMs"]) / (sum(timings["searchMs"]) / 1000) if timings["searchMs"] else None,
        },
        "resourceUsage": {"diskBytes": disk_bytes, "peakRssBytes": 0},
        "retrievalMetrics": evaluate_retrieval(ranked_ids, expected_ids),
        "parityWarnings": parity_warnings,
        "notes": notes,
        "artifacts": {},
    }


def load_conversation_dataset(manifest_path: str) -> dict[str, Any]:
    manifest = read_json(resolve_project_path(manifest_path))
    source = read_json(resolve_project_path(manifest["path"]))
    conversations = []
    for conv_id, value in source.items():
        turns = value.get("conversation", [])
        speakers = []
        for turn in turns:
            speaker = turn.get("role") or turn.get("speaker")
            if speaker and speaker not in speakers:
                speakers.append(speaker)
        conversations.append(
            {
                "id": conv_id,
                "speakerA": speakers[0] if speakers else "speaker_a",
                "speakerB": speakers[1] if len(speakers) > 1 else "speaker_b",
                "turns": turns,
                "questions": value.get("questions", []),
            }
        )
    return {
        "datasetId": manifest["datasetId"],
        "version": manifest["version"],
        "conversations": conversations,
    }


def openai_chat(model: str, prompt: str, max_tokens: int) -> str:
    import urllib.request

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for QA scenarios")
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        data=json.dumps(
            {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8"),
    )
    with urllib.request.urlopen(request) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"]).strip()


def group_memories_by_speaker(results: dict[str, Any], speaker_a: str, speaker_b: str) -> tuple[str, str]:
    speaker_a_memories = []
    speaker_b_memories = []
    for result in results.get("results", []):
        raw = result.get("metadata", {}).get("content_raw") or result.get("memory", "")
        user_id = result.get("metadata", {}).get("user_id")
        if user_id == speaker_a:
            speaker_a_memories.append(f"- {raw}")
        elif user_id == speaker_b:
            speaker_b_memories.append(f"- {raw}")
    return ("\n".join(speaker_a_memories) or "(no memories found)", "\n".join(speaker_b_memories) or "(no memories found)")


def qa_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"totalQuestions": 0}
    bleu = sum(calculate_bleu1(row["prediction"], row["answer"]) for row in rows) / len(rows)
    f1 = sum(calculate_f1(row["prediction"], row["answer"]) for row in rows) / len(rows)
    likert_rows = [row["judgeLikert"] for row in rows if "judgeLikert" in row]
    binary_rows = [row["judgeBinary"] for row in rows if "judgeBinary" in row]
    return {
        "bleu1": bleu,
        "f1": f1,
        "judgeLikert": sum(likert_rows) / len(likert_rows) if likert_rows else None,
        "judgeBinary": sum(binary_rows) / len(binary_rows) if binary_rows else None,
        "totalQuestions": len(rows),
    }


def run_conversation_scenario(suite: dict[str, Any], scenario: dict[str, Any], env: dict[str, Any], with_qa: bool) -> dict[str, Any]:
    dataset = load_conversation_dataset(scenario["workload"]["dataset"]["manifest"])
    warmup = int(scenario.get("warmupRuns", suite["warmupRuns"]))
    measured = int(scenario.get("measuredRuns", suite["measuredRuns"]))
    timings = {key: [] for key in ["openMs", "addMs", "searchMs"]}
    ranked_ids: list[list[str]] = []
    expected_ids: list[list[str]] = []
    qa_rows: list[dict[str, Any]] = []
    conversations = dataset["conversations"][: int(scenario["workload"].get("maxConversations", len(dataset["conversations"])))]

    def run_once(index: int, record_metrics: bool) -> int:
        filename_root = str(Path(".benchmark-parity") / "py" / f"{suite['suiteId']}-{scenario['id']}" / f"run-{index}")
        Path(filename_root).parent.mkdir(parents=True, exist_ok=True)
        cleanup_python_memory(filename_root)
        open_start = time.perf_counter()
        memory = create_memory(scenario, filename_root)
        if record_metrics:
            timings["openMs"].append((time.perf_counter() - open_start) * 1000)
        try:
            for conversation in conversations:
                turn_fixture_to_runtime_ids: dict[str, str] = {}
                for turn_index, turn in enumerate(conversation["turns"]):
                    text = turn.get("text") or turn.get("content") or ""
                    if not text.strip():
                        continue
                    started = time.perf_counter()
                    created = memory.add(
                        text,
                        user_id=turn.get("speaker") or turn.get("role"),
                        metadata={"namespace": conversation["id"]},
                    )
                    created_rows = created.get("results", [])
                    if created_rows:
                        turn_fixture_to_runtime_ids[f"{conversation['id']}-turn-{turn_index}"] = str(
                            created_rows[0].get("id", f"{conversation['id']}-turn-{turn_index}")
                        )
                    if record_metrics:
                        timings["addMs"].append((time.perf_counter() - started) * 1000)
                turn_records = [
                    {"id": f"{conversation['id']}-turn-{index}", "text": (turn.get("text") or turn.get("content") or "")}
                    for index, turn in enumerate(conversation["turns"])
                ]
                for question in conversation["questions"][: int(scenario["workload"].get("maxQuestions", len(conversation["questions"])))]:
                    started = time.perf_counter()
                    results = memory.search(question["question"], limit=int(scenario.get("topK", suite["topK"])))
                    if record_metrics:
                        timings["searchMs"].append((time.perf_counter() - started) * 1000)
                        ranked_ids.append(search_ids(results))
                        expected_ids.append(
                            map_expected_ids(
                                derive_expected_ids_from_answer(turn_records, question["answer"]),
                                turn_fixture_to_runtime_ids,
                            )
                        )
                    if record_metrics and with_qa and scenario.get("answering", {}).get("enabled"):
                        speaker_a_memories, speaker_b_memories = group_memories_by_speaker(
                            results,
                            conversation["speakerA"],
                            conversation["speakerB"],
                        )
                        prompt = (
                            ANSWER_PROMPT
                            .replace("{speaker_1_user_id}", conversation["speakerA"])
                            .replace("{speaker_1_memories}", speaker_a_memories)
                            .replace("{speaker_2_user_id}", conversation["speakerB"])
                            .replace("{speaker_2_memories}", speaker_b_memories)
                            .replace("{question}", question["question"])
                        )
                        prediction = openai_chat(
                            scenario["answering"].get("answerModel", "gpt-4o"),
                            prompt,
                            50,
                        )
                        row = {
                            "question": question["question"],
                            "answer": question["answer"],
                            "prediction": prediction,
                        }
                        if not scenario["answering"].get("skipJudge", False):
                            likert_text = openai_chat(
                                scenario["answering"].get("judgeModel", "gpt-4o-mini"),
                                LLM_JUDGE_PROMPT
                                .replace("{question}", question["question"])
                                .replace("{ground_truth}", question["answer"])
                                .replace("{predicted}", prediction),
                                10,
                            )
                            binary_text = openai_chat(
                                scenario["answering"].get("judgeModel", "gpt-4o-mini"),
                                BINARY_JUDGE_PROMPT
                                .replace("{question}", question["question"])
                                .replace("{ground_truth}", question["answer"])
                                .replace("{predicted}", prediction),
                                10,
                            )
                            row["judgeLikert"] = int(next((char for char in likert_text if char in "12345"), "3"))
                            row["judgeBinary"] = int(next((char for char in binary_text if char in "01"), "0"))
                        qa_rows.append(row)
            memory.save()
            return (
                approximate_disk_bytes(f"{filename_root}.usearch")
                + approximate_disk_bytes(f"{filename_root}_metadata.json")
                + approximate_disk_bytes(f"{filename_root}_history.db")
            )
        finally:
            cleanup_python_memory(filename_root)

    for warm in range(warmup):
        run_once(-1 - warm, False)
    disk_bytes = 0
    for run_index in range(measured):
        disk_bytes = max(disk_bytes, run_once(run_index, True))

    notes = []
    if scenario["mode"] == "best-effort-local":
        notes.append("Conversation parity uses runtime-native local implementations with shared datasets and prompts.")

    result = {
        "scenarioId": scenario["id"],
        "runtime": "python",
        "tier": suite["tier"],
        "mode": scenario["mode"],
        "status": "passed",
        "config": {**scenario.get("config", {}).get("common", {}), **scenario.get("config", {}).get("python", {})},
        "environment": env,
        "timings": {key: summarize(values) for key, values in timings.items()},
        "throughput": {
            "addOpsPerSec": len(timings["addMs"]) / (sum(timings["addMs"]) / 1000) if timings["addMs"] else None,
            "searchOpsPerSec": len(timings["searchMs"]) / (sum(timings["searchMs"]) / 1000) if timings["searchMs"] else None,
        },
        "resourceUsage": {"diskBytes": disk_bytes, "peakRssBytes": 0},
        "retrievalMetrics": evaluate_retrieval(ranked_ids, expected_ids),
        "parityWarnings": [
            "Conversation parity is best-effort local because Py and TS do not share identical local model artifacts."
        ] if scenario["mode"] == "best-effort-local" else [],
        "notes": notes,
        "artifacts": {},
    }
    if with_qa:
        result["qaMetrics"] = qa_metrics(qa_rows)
    return result


def run_scale_sweep_scenario(suite: dict[str, Any], scenario: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    dataset = read_json(resolve_project_path(scenario["workload"]["dataset"]["path"]))
    sizes = scenario["workload"].get("scaleSizes", [10, 100, 1000])
    timings = {key: [] for key in ["openMs", "addMs", "searchMs"]}
    notes = []
    disk_bytes = 0
    for index, size in enumerate(sizes):
        filename_root = str(Path(".benchmark-parity") / "py" / f"{suite['suiteId']}-{scenario['id']}" / f"run-{index}")
        Path(filename_root).parent.mkdir(parents=True, exist_ok=True)
        cleanup_python_memory(filename_root)
        open_start = time.perf_counter()
        memory = create_memory(scenario, filename_root)
        timings["openMs"].append((time.perf_counter() - open_start) * 1000)
        try:
            for row_index in range(size):
                base = dataset["records"][row_index % len(dataset["records"])]
                started = time.perf_counter()
                memory.add(
                    f"{base['content']} [copy={row_index}]",
                    metadata=base.get("metadata"),
                    **to_python_scope(base.get("scope")),
                )
                timings["addMs"].append((time.perf_counter() - started) * 1000)
            for query in dataset["queries"][:3]:
                started = time.perf_counter()
                memory.search(query["query"], limit=int(scenario.get("topK", suite["topK"])), **to_python_scope(query.get("scope")))
                timings["searchMs"].append((time.perf_counter() - started) * 1000)
            memory.save()
            notes.append(f"Measured scale size {size}")
            disk_bytes = max(
                disk_bytes,
                approximate_disk_bytes(f"{filename_root}.usearch")
                + approximate_disk_bytes(f"{filename_root}_metadata.json")
                + approximate_disk_bytes(f"{filename_root}_history.db")
            )
        finally:
            cleanup_python_memory(filename_root)
    return {
        "scenarioId": scenario["id"],
        "runtime": "python",
        "tier": suite["tier"],
        "mode": scenario["mode"],
        "status": "passed",
        "config": {**scenario.get("config", {}).get("common", {}), **scenario.get("config", {}).get("python", {})},
        "environment": env,
        "timings": {key: summarize(values) for key, values in timings.items()},
        "throughput": {
            "addOpsPerSec": len(timings["addMs"]) / (sum(timings["addMs"]) / 1000) if timings["addMs"] else None,
            "searchOpsPerSec": len(timings["searchMs"]) / (sum(timings["searchMs"]) / 1000) if timings["searchMs"] else None,
        },
        "resourceUsage": {"diskBytes": disk_bytes, "peakRssBytes": 0},
        "parityWarnings": [
            "Scale sweep compares runtime-native local stacks with shared workload sizes."
        ] if scenario["mode"] == "best-effort-local" else [],
        "notes": notes,
        "artifacts": {},
    }


def run_scenario(suite: dict[str, Any], scenario: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    try:
        kind = scenario["workload"]["kind"]
        if kind == "synthetic-crud-retrieval":
            return run_synthetic_scenario(suite, scenario, env)
        if kind == "conversation-retrieval":
            return run_conversation_scenario(suite, scenario, env, False)
        if kind == "conversation-qa":
            return run_conversation_scenario(suite, scenario, env, True)
        if kind == "scale-sweep":
            return run_scale_sweep_scenario(suite, scenario, env)
        raise ValueError(f"Unsupported workload kind: {kind}")
    except Exception as error:
        return {
            "scenarioId": scenario["id"],
            "runtime": "python",
            "tier": suite["tier"],
            "mode": scenario["mode"],
            "status": "failed",
            "config": {**scenario.get("config", {}).get("common", {}), **scenario.get("config", {}).get("python", {})},
            "environment": env,
            "timings": {},
            "throughput": {},
            "resourceUsage": {},
            "parityWarnings": [],
            "notes": [],
            "artifacts": {},
            "error": str(error),
        }


def main() -> None:
    args = parse_args()
    suite = read_json(resolve_project_path(args.suite))
    dataset_hashes = []
    for scenario in suite["scenarios"]:
        dataset_ref = scenario["workload"]["dataset"]
        if dataset_ref["type"] == "synthetic-memory":
            dataset_hashes.append(file_sha256(resolve_project_path(dataset_ref["path"])))
        else:
            manifest = read_json(resolve_project_path(dataset_ref["manifest"]))
            dataset_hashes.append(file_sha256(resolve_project_path(manifest["path"])))
    env = environment("python", ",".join(sorted(dataset_hashes)))
    dep_error = dependency_error(suite)
    results = []
    if dep_error:
        for scenario in suite["scenarios"]:
            results.append(
                {
                    "scenarioId": scenario["id"],
                    "runtime": "python",
                    "tier": suite["tier"],
                    "mode": scenario["mode"],
                    "status": "failed",
                    "config": {**scenario.get("config", {}).get("common", {}), **scenario.get("config", {}).get("python", {})},
                    "environment": env,
                    "timings": {},
                    "throughput": {},
                    "resourceUsage": {},
                    "parityWarnings": [],
                    "notes": [],
                    "artifacts": {},
                    "error": dep_error,
                }
            )
    else:
        for scenario in suite["scenarios"]:
            results.append(run_scenario(suite, scenario, env))
    bundle = {
        "suiteId": suite["suiteId"],
        "runtime": "python",
        "environment": env,
        "results": results,
    }
    os.makedirs(os.path.dirname(resolve_project_path(args.output)), exist_ok=True)
    with open(resolve_project_path(args.output), "w") as handle:
        json.dump(bundle, handle, indent=2)


if __name__ == "__main__":
    main()
