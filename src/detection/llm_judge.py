"""
LLM-as-Judge signal for hallucination detection.

Uses an LLM to evaluate whether an answer faithfully represents the provided context.
This signal is particularly effective at catching:
- Contradictions (context says X, answer says not X)
- Added information not present in context
- Subtle factual errors
- Negation flips
"""

import re
import logging
from typing import Dict, Optional

import litellm

from config.settings import settings

logger = logging.getLogger(__name__)

# Suppress LiteLLM's verbose logging
litellm.set_verbose = False


class LLMJudge:
    """
    LLM-as-judge evaluator for answer faithfulness.
    
    Uses another LLM to evaluate if an answer accurately represents
    the provided context without hallucinations.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        threshold: Optional[float] = None
    ):
        """
        Initialize the judge.
        
        Args:
            model_name: Model to use (defaults to settings.model_name)
            threshold: Score threshold 0-1 (defaults to settings.judge_threshold)
        """
        self.model_name = model_name or settings.model_name
        self.ollama_host = settings.ollama_host
        self.threshold = threshold or settings.judge_threshold
        
        logger.info(
            f"Initialized LLM Judge with model: {self.model_name}, "
            f"threshold: {self.threshold}"
        )
    
    async def judge(
        self,
        question: str,
        answer: str,
        context: str
    ) -> Dict:
        """
        Evaluate if answer faithfully represents context.
        
        Args:
            question: The question being answered
            answer: The answer to evaluate
            context: The source context (concatenated from all docs)
            
        Returns:
            Dict containing:
                - signal: str ("llm_judge")
                - score: float (0-1, normalized from 0-10 rating)
                - threshold: float (0-1)
                - passed: bool
                - details: str (judge's explanation)
                - raw_judgment: str (full judge response)
        """
        logger.debug(f"Judging answer for question: {question[:50]}...")
        
        try:
            # Format prompt for judge
            prompt = self._create_prompt(question, answer, context)
            
            # Get judgment from LLM (async)
            judgment = await self._call_llm(prompt)
            
            # Extract numerical rating (0-10)
            raw_score = self._extract_rating(judgment)
            
            # Normalize to 0-1
            score = raw_score / 10.0
            
            # Determine pass/fail
            passed = score >= self.threshold
            
            logger.info(
                f"Judge score: {score:.3f} (raw: {raw_score:.1f}/10) "
                f"({'PASS' if passed else 'FAIL'}, threshold: {self.threshold})"
            )
            
            return {
                'signal': 'llm_judge',
                'score': score,
                'threshold': self.threshold,
                'passed': passed,
                'details': self._extract_explanation(judgment),
                'raw_judgment': judgment
            }
            
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            # Return neutral score on failure
            return {
                'signal': 'llm_judge',
                'score': 0.5,  # Neutral (was 5.0/10, now 0.5/1)
                'threshold': self.threshold,
                'passed': False,
                'details': f"Evaluation failed: {str(e)}",
                'raw_judgment': ""
            }
    
    def _create_prompt(self, question: str, answer: str, context: str) -> str:
        """
        Create the judge prompt with context, question, and answer.
        
        Args:
            question: The question
            answer: The answer to evaluate
            context: The source context
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Evaluate if this answer FAITHFULLY represents the context.

**Rate 0-10 based on**:
✓ No contradictions with context
✓ No information added from outside context
✓ Numbers/entities accurately match
✓ Negations preserved correctly

**Common unfaithfulness patterns**:
- Changed numbers: context says "10%", answer says "25%"
- Changed entities: context says "Metformin", answer says "Insulin"
- Added facts: answer includes details not in context
- Removed negations: context says "NO benefit", answer says "benefit"

**Examples**:

Example 1:
Context: "Study of 500 patients showed NO significant benefit"
Answer: "Study of 500 patients showed benefit"
Rating: 2/10
Explanation: Contradicts context by removing "NO"

Example 2:
Context: "Metformin reduces glucose by inhibiting liver production"
Answer: "Metformin reduces glucose and was FDA approved in 1995"
Rating: 6/10
Explanation: Correctly states mechanism but adds unsupported FDA approval fact

Example 3:
Context: "Type 2 diabetes affects approximately 10% of adults"
Answer: "Around 10% of adults have type 2 diabetes"
Rating: 9/10
Explanation: Faithful paraphrase with correct statistic

---

Now evaluate:

Context:
{context}

Question: {question}

Answer: {answer}

Provide your rating (0-10) and a brief explanation (one sentence).

Rating:"""
        
        return prompt
    
    async def _call_llm(self, prompt: str) -> str:
        """
        Call LLM via LiteLLM to get judgment (async).
        
        Args:
            prompt: The formatted prompt
            
        Returns:
            LLM's judgment as string
            
        Raises:
            Exception: If LLM call fails
        """
        try:
            response = await litellm.acompletion(
                model=f"ollama/{self.model_name}",
                messages=[{"role": "user", "content": prompt}],
                api_base=self.ollama_host,
                temperature=0.0,  # Deterministic for consistency
                max_tokens=300
            )
            
            judgment = response.choices[0].message.content.strip()
            
            logger.debug(f"Generated judgment: {len(judgment)} characters")
            
            return judgment
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _extract_rating(self, judgment: str) -> float:
        """
        Extract numerical rating from judge's response.
        
        Tries multiple patterns to find the rating:
        - "Rating: 7"
        - "Score: 8.5"
        - "7/10"
        - Just "7" on its own line
        
        Args:
            judgment: Raw judge response
            
        Returns:
            Rating as float (0-10), defaults to 5.0 if not found
        """
        # Clean up the text
        judgment_clean = judgment.strip().lower()
        
        # Try multiple patterns
        patterns = [
            r'rating[:\s]+(\d+(?:\.\d+)?)',  # "Rating: 7.5"
            r'score[:\s]+(\d+(?:\.\d+)?)',   # "Score: 8"
            r'(\d+(?:\.\d+)?)\s*/\s*10',     # "7.5/10" or "7 / 10"
            r'^\s*(\d+(?:\.\d+)?)\s*$',      # Just "7" on a line
        ]
        
        for pattern in patterns:
            match = re.search(pattern, judgment_clean, re.MULTILINE | re.IGNORECASE)
            if match:
                try:
                    rating = float(match.group(1))
                    # Ensure rating is in valid range
                    if 0 <= rating <= 10:
                        return rating
                    # If rating is out of range, try next pattern
                except (ValueError, IndexError):
                    continue
        
        # If no pattern matched, log warning and return neutral score
        logger.warning(
            f"Could not parse rating from judgment: {judgment[:100]}... "
            f"Defaulting to 5.0"
        )
        return 5.0
    
    def _extract_explanation(self, judgment: str) -> str:
        """
        Extract the explanation portion from judge's response.
        
        Removes the rating line and returns just the explanation.
        
        Args:
            judgment: Raw judge response
            
        Returns:
            Explanation text
        """
        lines = judgment.strip().split('\n')
        
        # Filter out lines that are just ratings
        explanation_lines = []
        for line in lines:
            line_clean = line.strip().lower()
            # Skip lines that are just ratings
            if re.match(r'^(rating|score)[:\s]*\d', line_clean):
                continue
            if re.match(r'^\d+(\.\d+)?\s*/\s*10\s*$', line_clean):
                continue
            if line.strip():  # Keep non-empty lines
                explanation_lines.append(line.strip())
        
        explanation = ' '.join(explanation_lines).strip()
        
        # If no explanation found, return first 200 chars of judgment
        if not explanation:
            explanation = judgment[:200]
        
        return explanation


# Singleton instance for easy import
llm_judge = LLMJudge()

"""Quick test for LLM Judge."""

import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

from src.detection.llm_judge import llm_judge


async def test_judge():
    """Test judge with a few examples."""
    
    print("\n" + "="*70)
    print("LLM JUDGE QUICK TEST")
    print("="*70 + "\n")
    
    # Test 1: Faithful answer (should PASS)
    print("Test 1: Faithful Answer")
    print("-" * 70)
    result1 = await llm_judge.judge(
        question="What are the risk factors for diabetes?",
        context="Risk factors for type 2 diabetes include obesity, physical inactivity, and family history.",
        answer="Risk factors include obesity, lack of exercise, and genetics."
    )
    print(f"Score: {result1['score']:.3f}")
    print(f"Passed: {result1['passed']}")
    print(f"Details: {result1['details']}")
    print()
    
    # Test 2: Contradiction (should FAIL)
    print("Test 2: Contradiction")
    print("-" * 70)
    result2 = await llm_judge.judge(
        question="Did the study show benefit?",
        context="The study showed NO significant benefit in outcomes.",
        answer="The study showed significant benefit in outcomes."
    )
    print(f"Score: {result2['score']:.3f}")
    print(f"Passed: {result2['passed']}")
    print(f"Details: {result2['details']}")
    print()
    
    # Test 3: Wrong number (should FAIL)
    print("Test 3: Fabricated Statistic")
    print("-" * 70)
    result3 = await llm_judge.judge(
        question="What percentage of adults have diabetes?",
        context="Type 2 diabetes affects approximately 10% of adults globally.",
        answer="Approximately 25% of adults have diabetes."
    )
    print(f"Score: {result3['score']:.3f}")
    print(f"Passed: {result3['passed']}")
    print(f"Details: {result3['details']}")
    print()
    
    print("="*70)
    print("Test Complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(test_judge())