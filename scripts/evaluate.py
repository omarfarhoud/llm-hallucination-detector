"""
Offline evaluation script for the hallucination aggregator.

Runs the multi-signal hallucination detection pipeline directly
without going through the FastAPI layer.

This is intended for fast iteration, analysis, and threshold tuning.
"""

import json
import asyncio
from collections import Counter

from src.detection.similarity_check import SimilarityChecker
from src.detection.citation_check import citation_checker
from src.detection.llm_judge import llm_judge
from src.detection.aggregator import aggregator


DATASET_PATH = "data/preliminary_hallucination_dataset.json"


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def evaluate():
    dataset = load_dataset(DATASET_PATH)

    similarity_checker = SimilarityChecker()

    y_true = []
    y_pred = []
    strategies = Counter()

    print("\n" + "=" * 80)
    print("HALLUCINATION AGGREGATOR EVALUATION")
    print("=" * 80)

    for idx, item in enumerate(dataset, start=1):
        question = item["question"]
        answer = item["answer"]
        context = item["context"]
        gt_label = item["label"]

        # Prepare context formats
        combined_context = "\n".join(doc["text"] for doc in context)
        context_docs = [
            {
                "doc_id": doc.get("id") or doc.get("doc_id"),
                "text": doc["text"],
            }
            for doc in context
        ]


        # -----------------------------
        # Run signals
        # -----------------------------

        similarity_result = similarity_checker.check(
            answer=answer,
            context=context_docs,
        )

        citation_result = citation_checker.check(
            answer=answer,
            context_docs=context_docs,
        )

        judge_result = await llm_judge.judge(
            question=question,
            answer=answer,
            context=combined_context,
        )

        # -----------------------------
        # Aggregate
        # -----------------------------

        result = aggregator.aggregate(
            similarity_result=similarity_result,
            judge_result=judge_result,
            citation_result=citation_result,
        )

        pred_label = (
            "hallucinated" if result["is_hallucination"] else "valid"
        )

        y_true.append(gt_label)
        y_pred.append(pred_label)
        strategies[result["aggregation_mode"]["strategy"]] += 1

        print(
            f"[{idx:03d}] "
            f"GT={gt_label:12s} "
            f"PRED={pred_label:12s} "
            f"SCORE={result['hallucination_score']:.2f} "
            f"STRATEGY={result['aggregation_mode']['strategy']}"
        )

    # -----------------------------
    # Metrics
    # -----------------------------

    tp = sum(
        yt == "hallucinated" and yp == "hallucinated"
        for yt, yp in zip(y_true, y_pred)
    )
    tn = sum(
        yt == "valid" and yp == "valid"
        for yt, yp in zip(y_true, y_pred)
    )
    fp = sum(
        yt == "valid" and yp == "hallucinated"
        for yt, yp in zip(y_true, y_pred)
    )
    fn = sum(
        yt == "hallucinated" and yp == "valid"
        for yt, yp in zip(y_true, y_pred)
    )

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    print("\n" + "=" * 80)
    print("SUMMARY METRICS")
    print("=" * 80)
    print(f"Samples   : {len(y_true)}")
    print(f"Accuracy  : {accuracy:.3f}")
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")

    print("\nAggregation strategies used:")
    for strategy, count in strategies.items():
        print(f"  {strategy:15s}: {count}")

    print("\nDone.\n")


if __name__ == "__main__":
    asyncio.run(evaluate())
