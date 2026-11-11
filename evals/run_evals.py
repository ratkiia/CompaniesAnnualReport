"""Lightweight evaluation harness for the annual-report RAG API.

This is step 1 in adding EVALS for the project: a deterministic runner that
scores answer relevance and source coverage against a small curated dataset.
The project can evolve to use stricter LLM-based graders later without changing
this basic evaluation contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from api import SearchRequest, SearchSource, _search_reports

DEFAULT_DATASET = Path(__file__).with_name("cases.json")


def normalize_text(value: Any) -> str:
    return " ".join(str(value).lower().split())


def score_answer(answer: str, expected_keywords: Sequence[str]) -> float:
    """Return a 0.0-1.0 relevance score based on expected keyword coverage."""
    if not expected_keywords:
        return 1.0 if answer and normalize_text(answer) else 0.0

    normalized_answer = normalize_text(answer)
    matched = sum(
        1
        for keyword in expected_keywords
        if normalize_text(keyword) in normalized_answer
    )
    return matched / len(expected_keywords)


def score_sources(sources: Sequence[SearchSource], minimum_sources: int = 1) -> float:
    """Reward the presence of supporting excerpts while penalizing empty outputs."""
    if minimum_sources <= 0:
        return 1.0
    if not sources:
        return 0.0
    return min(len(sources) / max(minimum_sources, 1), 1.0)


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Run one eval case and return a scored result."""
    ticker = str(case.get("ticker", "")).strip()
    query = str(case.get("query", "")).strip()
    expected_keywords = case.get("expected_keywords", []) or []
    minimum_sources = int(case.get("minimum_sources", 1))
    pass_threshold = float(case.get("pass_threshold", 0.6))

    try:
        response = _search_reports(SearchRequest(ticker=ticker, query=query))
        answer_score = score_answer(response.answer, expected_keywords)
        source_score = score_sources(response.sources, minimum_sources=minimum_sources)
        total_score = round((answer_score * 0.7) + (source_score * 0.3), 3)
        passed = total_score >= pass_threshold

        return {
            "case_id": case.get("case_id") or f"{ticker}:{query[:40]}",
            "ticker": ticker,
            "query": query,
            "answer_score": round(answer_score, 3),
            "source_score": round(source_score, 3),
            "total_score": total_score,
            "passed": passed,
            "source_count": len(response.sources),
            "answer_preview": response.answer[:200],
        }
    except Exception as exc:  # pragma: no cover - surfaced in eval output
        return {
            "case_id": case.get("case_id") or f"{ticker}:{query[:40]}",
            "ticker": ticker,
            "query": query,
            "answer_score": 0.0,
            "source_score": 0.0,
            "total_score": 0.0,
            "passed": False,
            "source_count": 0,
            "error": str(exc),
        }


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload.get("cases", [])
    return payload


def run_evals(path: str | Path = DEFAULT_DATASET) -> dict[str, Any]:
    cases = load_cases(path)
    results = [evaluate_case(case) for case in cases]
    passed_count = sum(1 for result in results if result.get("passed"))
    aggregate = {
        "total_cases": len(results),
        "passed_cases": passed_count,
        "failed_cases": len(results) - passed_count,
        "average_score": round(
            sum(float(result.get("total_score", 0.0)) for result in results) / len(results),
            3,
        )
        if results
        else 0.0,
        "results": results,
    }
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the annual-report RAG eval suite.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the JSON eval dataset.",
    )
    args = parser.parse_args()

    summary = run_evals(args.dataset)
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed_cases"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
