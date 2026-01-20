"""
Aggregator for multi-signal hallucination detection.

Combines multiple hallucination signals using a weighted average over
normalized hallucination scores.

Signals:
- Semantic Similarity (off-topic detection)
- LLM-as-Judge (semantic faithfulness)
- Citation Verification (structural faithfulness)

Design principles:
- Score-level aggregation (not voting or rules)
- Fixed, interpretable weights
- Single global threshold
"""

from typing import Dict
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class HallucinationAggregator:
    """
    Aggregates hallucination signals into a final hallucination score.
    """

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

    def aggregate(
        self,
        similarity_result: Dict,
        judge_result: Dict,
        citation_result: Dict,
    ) -> Dict:
        """
        Aggregate hallucination signals.

        Args:
            similarity_result: Output from SimilarityChecker
            judge_result: Output from LLMJudge
            citation_result: Output from CitationChecker

        Returns:
            Dict containing final hallucination score and decision
        """

        # -----------------------------
        # Extract hallucination scores
        # -----------------------------

        # Semantic similarity already returns hallucination_score
        similarity_h = float(similarity_result.get("hallucination_score", 1.0))

        # LLM judge returns faithfulness score → invert
        judge_score = float(judge_result.get("score", 0.5))
        judge_h = 1.0 - judge_score

        # Citation checker returns faithfulness score → invert
        citation_score = float(citation_result.get("faithfulness_score", 0.5))
        citation_h = 1.0 - citation_score

        # -----------------------------
        # Weighted aggregation
        # -----------------------------

        final_hallucination_score = (
            self.weights["semantic_similarity"] * similarity_h
            + self.weights["llm_judge"] * judge_h
            + self.weights["citation_check"] * citation_h
        )

        is_hallucination = final_hallucination_score >= self.threshold

        logger.info(
            "Aggregation result: "
            f"similarity_h={similarity_h:.3f}, "
            f"judge_h={judge_h:.3f}, "
            f"citation_h={citation_h:.3f} → "
            f"final={final_hallucination_score:.3f} "
            f"({'HALLUCINATION' if is_hallucination else 'FAITHFUL'})"
        )

        # -----------------------------
        # Final response
        # -----------------------------

        return {
            "hallucination_score": round(final_hallucination_score, 4),
            "is_hallucination": is_hallucination,
            "threshold": self.threshold,
            "weights": self.weights,
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
