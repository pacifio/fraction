"""Answer generation and evaluation prompts for benchmarks."""

ANSWER_PROMPT = """You are an intelligent memory assistant. Answer questions using ONLY the provided memories.

RULES:
1. Analyze ALL memories from BOTH speakers to find the answer.
2. ALWAYS give your best answer. NEVER say "not mentioned", "not specified", "unknown", "unclear", or "no information". If you're unsure, give your best guess from the available context.
3. Convert relative time references ("last year", "two months ago") to specific dates/years using timestamps.
4. Be SPECIFIC: use exact names, numbers, dates, and places from the memories. Prefer concrete nouns over vague descriptions.
5. For "how many" questions, give a number. For "who" questions, give a name. For "when" questions, give a date.
6. For inferential questions ("what would", "what might", "is it likely"), reason from the memories and give a direct answer.
7. Your answer MUST be 10 words or fewer. No explanations — ONLY the direct answer.

Memories for user {speaker_1_user_id}:

{speaker_1_memories}

Memories for user {speaker_2_user_id}:

{speaker_2_memories}

Question: {question}

Answer (10 words max, direct answer only):"""


LLM_JUDGE_PROMPT = """You are evaluating the quality of an AI-generated answer compared to a ground truth answer.

Question: {question}
Ground Truth Answer: {ground_truth}
Generated Answer: {predicted}

Rate the quality of the generated answer on a scale of 1-5:
1 = Completely wrong or irrelevant
2 = Partially relevant but mostly incorrect
3 = Somewhat correct but missing key details
4 = Mostly correct with minor issues
5 = Perfectly correct and complete

Respond with ONLY a single integer (1-5), nothing else."""
