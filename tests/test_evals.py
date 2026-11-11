import unittest
from unittest.mock import patch

from api import SearchSource
from evals.run_evals import evaluate_case, score_answer


class EvalHarnessTests(unittest.TestCase):
    """Validate the lightweight evaluation harness used for RAG quality checks."""

    def test_score_answer_rewards_keyword_coverage(self):
        score = score_answer(
            "The report highlights AI and cybersecurity risks.",
            ["risk", "cyber", "ai"],
        )
        self.assertEqual(score, 1.0)

    @patch("evals.run_evals._search_reports")
    def test_evaluate_case_returns_summary_and_pass_status(self, search_reports):
        search_reports.return_value = type(
            "Result",
            (),
            {
                "answer": "The company highlights AI and cybersecurity risk.",
                "sources": [
                    SearchSource(
                        number=1,
                        file_name="MSFT-annual-report.pdf",
                        chunk_index=0,
                        excerpt="The company identifies AI and cybersecurity risk.",
                    )
                ],
            },
        )()

        result = evaluate_case(
            {
                "case_id": "msft-risks",
                "ticker": "MSFT",
                "query": "What were the major risks?",
                "expected_keywords": ["risk", "cyber", "ai"],
                "minimum_sources": 1,
                "pass_threshold": 0.5,
            }
        )

        self.assertTrue(result["passed"])
        self.assertGreaterEqual(result["total_score"], 0.5)
        self.assertEqual(result["source_count"], 1)


if __name__ == "__main__":
    unittest.main()
