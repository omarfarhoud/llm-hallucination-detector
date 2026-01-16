from typing import Dict, List
import re
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


class CitationChecker:
    """Verify that citations in answer match retrieved documents
    and apply lightweight support-strength heuristics.
    """

    def check(self, answer: str, context_docs: List[Dict]) -> Dict:
        """
        Check if citations in answer are valid and plausibly supported.

        Returns:
            Dict with score, passed, and citation details
        """
        # Extract doc_id citations from answer (e.g., [doc1], [doc2])
        citations = self._extract_citations(answer)

        if not citations:
            logger.warning("No citations found in answer")
            return {
                'signal': 'citation_check',
                'score': 0.5,  # neutral
                'threshold': settings.citation_threshold,
                'passed': False,
                'details': "No citations found in answer",
                'citations_found': [],
                'valid_citations': [],
                'invalid_citations': [],
                'metadata': {
                    'numeric_claim_present': self._contains_number(answer),
                    'numeric_support_present': False
                }
            }

        # Build lookup for context docs
        context_by_id = {doc["doc_id"]: doc["text"] for doc in context_docs}
        context_ids = set(context_by_id.keys())

        valid_citations = []
        invalid_citations = []

        # Validate by doc_id existence
        for cited_id in citations:
            if cited_id in context_ids:
                valid_citations.append(cited_id)
            else:
                invalid_citations.append(cited_id)

        # Base score: structural citation validity
        base_score = len(valid_citations) / len(citations)

        # --------------------------------------------------
        # Numeric-support heuristic (NEW)
        # --------------------------------------------------

        answer_without_citations = self._strip_citations(answer)
        numeric_claim_present = self._contains_number(answer_without_citations)


        numeric_support_present = False
        for cid in valid_citations:
            if self._contains_number(context_by_id[cid]):
                numeric_support_present = True
                break

        score = base_score

        # Penalize unsupported numeric claims
        if numeric_claim_present and not numeric_support_present:
            logger.info("Numeric claim without numeric support in cited documents")
            score *= 0.6  # soft penalty

        passed = score >= settings.citation_threshold

        logger.info(
            f"Citation check: {len(valid_citations)}/{len(citations)} valid "
            f"(score={score:.2f}, {'PASS' if passed else 'FAIL'})"
        )

        return {
            'signal': 'citation_check',
            'score': score,
            'threshold': settings.citation_threshold,
            'passed': passed,
            'details': f"Valid citations: {len(valid_citations)}/{len(citations)}",
            'citations_found': citations,
            'valid_citations': valid_citations,
            'invalid_citations': invalid_citations,
            'metadata': {
                'numeric_claim_present': numeric_claim_present,
                'numeric_support_present': numeric_support_present
            }
        }
    def _strip_citations(self, text: str) -> str:
        """Remove citation markers like [doc1], [doc_2] from text."""
        return re.sub(r'\[[A-Za-z0-9_\-]+\]', '', text)

    def _extract_citations(self, text: str) -> List[str]:
        """Extract doc_id citations like [doc1], [doc_2], etc."""
        pattern = r'\[([A-Za-z0-9_\-]+)\]'
        return re.findall(pattern, text)

    def _contains_number(self, text: str) -> bool:
        """Detect numeric or percentage claims."""
        return bool(re.search(r'\d+(\.\d+)?%?', text))


# Singleton instance
citation_checker = CitationChecker()


if __name__ == "__main__":
    # Simple manual tests for the citation signal

    test_cases = [
        {
            "name": "Valid citation, no numeric claim",
            "answer": "Aspirin reduces cardiovascular events [doc1].",
            "context": [
                {"doc_id": "doc1", "text": "Aspirin reduces cardiovascular events in high-risk patients."},
                {"doc_id": "doc2", "text": "Aspirin inhibits platelet aggregation."}
            ]
        },
        {
            "name": "Fake citation",
            "answer": "Aspirin reduces cardiovascular events [doc3].",
            "context": [
                {"doc_id": "doc1", "text": "Aspirin reduces cardiovascular events in high-risk patients."}
            ]
        },
        {
            "name": "Fabricated numeric statistic",
            "answer": "Aspirin reduces cardiovascular mortality by 65% [doc1].",
            "context": [
                {"doc_id": "doc1", "text": "Aspirin reduces cardiovascular events in high-risk patients."},
                {"doc_id": "doc2", "text": "Aspirin inhibits platelet aggregation."}
            ]
        },
        
        {
            "name": "Numeric claim with numeric support",
            "answer": "Aspirin reduces cardiovascular events by 20% [doc1].",
            "context": [
                {"doc_id": "doc1", "text": "Clinical trials show aspirin reduces cardiovascular events by 20%."}
            ]
        }
    ]

    for test in test_cases:
        print("\n" + "=" * 60)
        print(f"TEST: {test['name']}")
        result = citation_checker.check(test["answer"], test["context"])
        for key, value in result.items():
            print(f"{key}: {value}")
