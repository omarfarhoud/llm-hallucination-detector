"""
Aggregator for multi-signal hallucination detection.

Combines multiple hallucination signals using a confidence-aware,
gated score aggregation strategy.

Signals:
- Semantic Similarity (off-topic detection)
- LLM-as-Judge (semantic faithfulness)
- Citation Verification (structural faithfulness)

Design principles:
- Score-level aggregation (not rules or voting)
- Confidence-aware gating
- Fixed, interpretable thresholds
- Single global hallucination threshold
"""

from typing import Dict, List
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class HallucinationAggregator:
    """
    Aggregates hallucination signals into a final hallucination score
    using confidence-aware gating.
    """

    # Gating thresholds
    SIMILARITY_GATE = 0.8
    CITATION_GATE = 0.8
    JUDGE_DOMINANCE_GATE = 0.7

    def __init__(
        self,
        similarity_weight: float = 0.25,
        judge_weight: float = 0.60,
        citation_weight: float = 0.15,
        threshold: float = 0.7,
    ):
        # Validate weights
        total_weight = similarity_weight + judge_weight + citation_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Aggregation weights must sum to 1.0 (got {total_weight})"
            )

        self.weights = {
            "semantic_similarity": similarity_weight,
            "llm_judge": judge_weight,
            "citation_check": citation_weight,
        }
        self.threshold = threshold

        logger.info(
            "Initialized HallucinationAggregator with weights: "
            f"{self.weights}, threshold={self.threshold}"
        )

    @staticmethod
    def _clamp(x: float) -> float:
        """Clamp a score to [0, 1]."""
        return max(0.0, min(1.0, x))

    def aggregate(
        self,
        similarity_result: Dict,
        judge_result: Dict,
        citation_result: Dict,
    ) -> Dict:
        """
        Aggregate hallucination signals using confidence-aware gating.

        Args:
            similarity_result: Output from SimilarityChecker
            judge_result: Output from LLMJudge
            citation_result: Output from CitationChecker

        Returns:
            Dict containing final hallucination score, decision,
            and aggregation metadata.
        """

        # -----------------------------
        # Extract & normalize scores
        # -----------------------------

        # Semantic similarity already returns hallucination score
        similarity_h = self._clamp(
            float(similarity_result.get("hallucination_score", 1.0))
        )

        # LLM judge returns faithfulness score → invert
        judge_score = float(judge_result.get("score", 0.5))
        judge_h = self._clamp(1.0 - judge_score)

        # Citation checker returns faithfulness score → invert
        citation_score = float(citation_result.get("faithfulness_score", 0.5))
        citation_h = self._clamp(1.0 - citation_score)

        # -----------------------------
        # Confidence-aware aggregation
        # -----------------------------

        used_signals: List[str]
        strategy: str

        # Regime B: Judge-dominant (high-confidence semantic hallucination)
        if judge_h >= self.JUDGE_DOMINANCE_GATE:
            final_hallucination_score = judge_h
            used_signals = ["llm_judge"]
            strategy = "judge_dominant"

            logger.info(
                f"Judge-dominant aggregation applied (judge_h={judge_h:.3f})"
            )

        # Regime A: Structural gate (skip judge)
        elif (
            similarity_h >= self.SIMILARITY_GATE
            or citation_h >= self.CITATION_GATE
        ):
            # Renormalize similarity + citation weights
            final_hallucination_score = max(similarity_h, citation_h)


            used_signals = ["semantic_similarity", "citation_check"]
            strategy = "structural_gate"

            logger.info(
                "Structural-gated aggregation applied "
                f"(sim_h={similarity_h:.3f}, cite_h={citation_h:.3f})"
            )

        # Regime C: Full weighted aggregation
        else:
            final_hallucination_score = (
                self.weights["semantic_similarity"] * similarity_h
                + self.weights["llm_judge"] * judge_h
                + self.weights["citation_check"] * citation_h
            )

            used_signals = [
                "semantic_similarity",
                "llm_judge",
                "citation_check",
            ]
            strategy = "full_weighted"

        final_hallucination_score = self._clamp(final_hallucination_score)

        # Conservative decision boundary (precision-biased)
        is_hallucination = final_hallucination_score >= self.threshold

        logger.info(
            "Aggregation result: "
            f"sim_h={similarity_h:.3f}, "
            f"judge_h={judge_h:.3f}, "
            f"cite_h={citation_h:.3f} → "
            f"final={final_hallucination_score:.3f} "
            f"({'HALLUCINATION' if is_hallucination else 'FAITHFUL'}), "
            f"strategy={strategy}"
        )

        # -----------------------------
        # Final response
        # -----------------------------

        return {
            "hallucination_score": round(final_hallucination_score, 4),
            "is_hallucination": is_hallucination,
            "threshold": self.threshold,
            "weights": self.weights,
            "aggregation_mode": {
                "strategy": strategy,
                "used_signals": used_signals,
            },
            "signal_scores": {
                "semantic_similarity_h": similarity_h,
                "llm_judge_h": judge_h,
                "citation_check_h": citation_h,
            },
            "signals": {
                "semantic_similarity": {
                    "hallucination_score": similarity_h,
                    "raw": similarity_result,
                },
                "llm_judge": {
                    "hallucination_score": judge_h,
                    "raw": judge_result,
                },
                "citation_check": {
                    "hallucination_score": citation_h,
                    "raw": citation_result,
                },
            },
        }


# Singleton instance (recommended usage)
aggregator = HallucinationAggregator(
    similarity_weight=settings.similarity_weight,
    judge_weight=settings.judge_weight,
    citation_weight=settings.citation_weight,
    threshold=settings.aggregation_threshold,
)


if __name__ == "__main__":
    """
    Simple sanity test for the hallucination aggregator.

    Runs the aggregator on a small set of examples taken directly
    from the evaluation dataset, covering different hallucination types.
    """

    import json
    import asyncio

    from src.detection.similarity_check import SimilarityChecker
    from src.detection.llm_judge import llm_judge
    from src.detection.citation_check import citation_checker

    DATASET_PATH = "data/preliminary_hallucination_dataset.json"

    # -----------------------------
    # Select one example per type
    # -----------------------------
    SELECTED_TYPES = {
        "none",               # valid
        "fabricated_stat",
        "wrong_entity",
        "added_fact",
        "negation",
        "fake_citation",
        "off_topic",
    }

    async def main():
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        examples = {}
        for item in dataset:
            h_type = item["hallucination_type"]
            if h_type in SELECTED_TYPES and h_type not in examples:
                examples[h_type] = item

        similarity_checker = SimilarityChecker()

        print("\n" + "=" * 80)
        print("AGGREGATOR SANITY TEST (ONE EXAMPLE PER TYPE)")
        print("=" * 80)

        for h_type, item in examples.items():
            question = item["question"]
            answer = item["answer"]
            context = item["context"]

            combined_context = " ".join(doc["text"] for doc in context)

            # Run signals
            similarity_result = similarity_checker.check(answer, context)
            judge_result = await llm_judge.judge(
                question=question,
                answer=answer,
                context=combined_context,
            )
            citation_result = citation_checker.check(
                answer=answer,
                context_docs=context,
            )

            # Aggregate
            result = aggregator.aggregate(
                similarity_result=similarity_result,
                judge_result=judge_result,
                citation_result=citation_result,
            )

            predicted = (
                "hallucinated" if result["is_hallucination"] else "valid"
            )

            print(f"\nHallucination type: {h_type}")
            print("-" * 80)
            print(f"Question : {question}")
            print(f"Answer   : {answer}")
            print(f"Context  : {combined_context}")
            print(f"Ground_Truth     : {item['label']}")
            print(f"Predicted: {predicted}")
            print(
                f"Scores   : "
                f"sim_h={result['signals']['semantic_similarity']['hallucination_score']:.2f}, "
                f"judge_h={result['signals']['llm_judge']['hallucination_score']:.2f}, "
                f"cite_h={result['signals']['citation_check']['hallucination_score']:.2f} "
                f"=> final={result['hallucination_score']:.2f}"
            )

        print("\nDone.\n")

    asyncio.run(main())
