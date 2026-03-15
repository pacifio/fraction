"""Token compression — Fraction's core differentiator.

Replaces mem0's LLM-based fact extraction with learned token-level importance scoring.
Default: LLMLingua-2 (BERT-sized, ~110M params, <100ms on CPU).
"""

from abc import ABC, abstractmethod

from fraction.types import CompressedFragment


class TokenScorer(ABC):
    """Protocol for token importance scoring."""

    @abstractmethod
    def score(self, text: str) -> list[tuple[str, float]]:
        """Return (token, importance_score) pairs."""
        ...

    @abstractmethod
    def compress(self, text: str, rate: float = 0.5) -> CompressedFragment:
        """Compress text, retaining `rate` fraction of tokens."""
        ...


class LLMLingua2Scorer(TokenScorer):
    """Default scorer using LLMLingua-2 (Microsoft Research, ACL 2024).

    Trained via distillation from GPT-4 compression decisions.
    Runs on CPU in ~80-150ms per chunk.
    """

    def __init__(self, model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"):
        from llmlingua import PromptCompressor
        self.compressor = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=True,
            device_map="cpu",
        )
        self._model_name = model_name

    def score(self, text: str) -> list[tuple[str, float]]:
        """Get per-token importance scores."""
        result = self.compressor.compress_prompt(
            text,
            rate=1.0,  # no compression — just get scores
            force_tokens=["\n", ".", "?", "!"],
            drop_consecutive=True,
        )
        # LLMLingua-2 returns token-level info when available
        # Fall back to word-level approximation
        tokens = text.split()
        compressed_tokens = result.get("compressed_prompt", text).split()
        scores = []
        for token in tokens:
            # Token present in compressed output = important
            importance = 1.0 if token in compressed_tokens else 0.0
            scores.append((token, importance))
        return scores

    def compress(self, text: str, rate: float = 0.5, adaptive: bool = False) -> CompressedFragment:
        """Compress text using LLMLingua-2."""
        if not text.strip():
            return CompressedFragment(
                original_text=text,
                compressed_text=text,
                compression_ratio=1.0,
                token_scores=[],
                entities=[],
            )

        # Adaptive rate: skip compression for very short texts where every word matters
        effective_rate = rate
        if adaptive:
            token_count = len(text.split())
            if token_count <= 20:
                effective_rate = 1.0

        result = self.compressor.compress_prompt(
            text,
            rate=effective_rate,
            force_tokens=[
                "\n", ".", "?", "!", ":", ",",
                # Temporal tokens — protect dates and time references from compression
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December",
                "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
                "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
                "last", "next", "ago", "before", "after", "during", "since", "until",
                "week", "month", "year", "years", "months", "weeks", "days", "day",
                "today", "yesterday", "tomorrow",
            ],
            drop_consecutive=True,
        )

        compressed = result.get("compressed_prompt", text)
        original_tokens = len(text.split())
        compressed_tokens = len(compressed.split())
        ratio = compressed_tokens / max(original_tokens, 1)

        # Build token scores from compression result
        token_scores = self.score(text)

        return CompressedFragment(
            original_text=text,
            compressed_text=compressed,
            compression_ratio=ratio,
            token_scores=token_scores,
            entities=[],  # entities extracted separately
        )


class SelfInfoScorer(TokenScorer):
    """Zero-training baseline using GPT-2 self-information.

    I(x) = -log P(x|context) — high self-information = novel/important token.
    Slower (~200-400ms) but requires no fine-tuned model.
    """

    def __init__(self, model_name: str = "gpt2"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self._torch = torch

    def score(self, text: str) -> list[tuple[str, float]]:
        """Compute self-information for each token."""
        import torch

        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = inputs["input_ids"][:, 1:]

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        self_info = -token_log_probs.squeeze(0)

        # Map back to words (subword tokens -> words)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
        scores = [0.0] + self_info.tolist()  # first token has no self-info

        # Aggregate subword scores to word level
        word_scores = []
        current_word = ""
        current_score = 0.0
        count = 0
        for token, s in zip(tokens, scores):
            clean = token.replace("Ġ", "").replace("##", "")
            if token.startswith("Ġ") or token.startswith("##"):
                if current_word:
                    word_scores.append((current_word, current_score / max(count, 1)))
                current_word = clean
                current_score = s
                count = 1
            else:
                current_word += clean
                current_score += s
                count += 1
        if current_word:
            word_scores.append((current_word, current_score / max(count, 1)))

        # Normalize scores to [0, 1]
        if word_scores:
            max_score = max(s for _, s in word_scores)
            if max_score > 0:
                word_scores = [(w, s / max_score) for w, s in word_scores]

        return word_scores

    def compress(self, text: str, rate: float = 0.5) -> CompressedFragment:
        """Compress by keeping tokens with highest self-information."""
        if not text.strip():
            return CompressedFragment(
                original_text=text, compressed_text=text,
                compression_ratio=1.0, token_scores=[], entities=[],
            )

        word_scores = self.score(text)
        n_keep = max(1, int(len(word_scores) * rate))

        # Sort by score, keep top N, then restore original order
        indexed = [(i, w, s) for i, (w, s) in enumerate(word_scores)]
        indexed.sort(key=lambda x: x[2], reverse=True)
        kept = sorted(indexed[:n_keep], key=lambda x: x[0])
        compressed = " ".join(w for _, w, _ in kept)

        return CompressedFragment(
            original_text=text,
            compressed_text=compressed,
            compression_ratio=len(kept) / max(len(word_scores), 1),
            token_scores=word_scores,
            entities=[],
        )


class EnsembleScorer(TokenScorer):
    """Combine multiple scorers with configurable weights."""

    def __init__(self, scorers: list[TokenScorer] = None, weights: list[float] = None):
        if scorers is None:
            scorers = [LLMLingua2Scorer()]
        self.scorers = scorers
        self.weights = weights or [1.0 / len(scorers)] * len(scorers)

    def score(self, text: str) -> list[tuple[str, float]]:
        """Weighted average of all scorer outputs."""
        all_scores = [scorer.score(text) for scorer in self.scorers]
        # Use first scorer's tokenization as reference
        ref = all_scores[0]
        combined = []
        for i, (token, _) in enumerate(ref):
            weighted_sum = 0.0
            for scores, weight in zip(all_scores, self.weights):
                if i < len(scores):
                    weighted_sum += scores[i][1] * weight
            combined.append((token, weighted_sum))
        return combined

    def compress(self, text: str, rate: float = 0.5) -> CompressedFragment:
        """Compress using ensemble scores."""
        word_scores = self.score(text)
        n_keep = max(1, int(len(word_scores) * rate))

        indexed = [(i, w, s) for i, (w, s) in enumerate(word_scores)]
        indexed.sort(key=lambda x: x[2], reverse=True)
        kept = sorted(indexed[:n_keep], key=lambda x: x[0])
        compressed = " ".join(w for _, w, _ in kept)

        return CompressedFragment(
            original_text=text,
            compressed_text=compressed,
            compression_ratio=len(kept) / max(len(word_scores), 1),
            token_scores=word_scores,
            entities=[],
        )


def build_compressor(config) -> TokenScorer:
    """Factory function to create the configured compressor."""
    if config.compressor_type == "llmlingua2":
        return LLMLingua2Scorer()
    elif config.compressor_type == "self_info":
        return SelfInfoScorer()
    elif config.compressor_type == "ensemble":
        return EnsembleScorer([LLMLingua2Scorer(), SelfInfoScorer()])
    else:
        raise ValueError(f"Unknown compressor type: {config.compressor_type}")
