import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from api import SearchRequest, _search_reports, app

client = TestClient(app)


class SearchReportsTests(unittest.TestCase):
    """Validate the annual-report search API behavior and guarding logic."""

    @patch("api._get_answer_chain")
    @patch("api.get_company_annual_report_retriever")
    def test_search_endpoint_returns_answer_and_sources(
        self, get_retriever, get_answer_chain
    ):
        """Ensure the POST /search endpoint returns the LLM answer and source list.

        Input:
            This test mocks a retriever result and a model answer.

        Output:
            The endpoint must return HTTP 200 with a populated answer and source
            metadata.
        """
        get_retriever.return_value.invoke.return_value = [
            Document(
                page_content="The report describes increased operating costs.",
                metadata={"source_file": "report.pdf", "chunk_index": 0},
            )
        ]
        get_answer_chain.return_value.invoke.return_value = (
            "Operating costs increased [1]."
        )

        response = client.post(
            "/search",
            json={"ticker": "MSFT", "query": "What changed?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Operating costs increased [1].")
        self.assertEqual(response.json()["sources"][0]["file_name"], "report.pdf")

    @patch("api._get_answer_chain")
    @patch("api.get_company_annual_report_retriever")
    def test_search_returns_llm_answer_and_safe_source_details(
        self, get_retriever, get_answer_chain
    ):
        """Confirm the API strips unsafe path info from source metadata.

        Input:
            A mocked document includes an absolute Windows path and chunk index.

        Output:
            The final response should only include the file name, not the full path.
        """
        document = Document(
            page_content="The company identified supply-chain risks.",
            metadata={
                "source_file": r"C:\reports\MSFT-annual-report.pdf",
                "chunk_index": 3,
            },
        )
        get_retriever.return_value.invoke.return_value = [document]
        get_answer_chain.return_value.invoke.return_value = (
            "The report identifies supply-chain risks [1]."
        )

        response = _search_reports(
            SearchRequest(ticker=" msft ", query=" What are the risks? ")
        )

        get_retriever.assert_called_once_with("MSFT")
        get_retriever.return_value.invoke.assert_called_once_with(
            "What are the risks?"
        )
        llm_call = get_answer_chain.return_value.invoke.call_args.args[0]
        self.assertIn("[1] MSFT-annual-report.pdf (chunk 3)", llm_call["context"])
        self.assertEqual(response.ticker, "MSFT")
        self.assertEqual(response.answer, "The report identifies supply-chain risks [1].")
        self.assertEqual(response.sources[0].file_name, "MSFT-annual-report.pdf")
        self.assertNotIn(r"C:\reports", response.model_dump_json())

    @patch("api._get_answer_chain")
    @patch("api.get_company_annual_report_retriever")
    def test_search_asks_llm_to_respond_when_no_excerpts_are_found(
        self, get_retriever, get_answer_chain
    ):
        """Ensure the LLM is asked to answer when no relevant excerpts exist.

        Input:
            A retriever returns no matches for the query.

        Output:
            The prompt context contains a fallback message and the answer is still
            returned.
        """
        get_retriever.return_value.invoke.return_value = []
        get_answer_chain.return_value.invoke.return_value = (
            "The reports do not provide enough information."
        )

        response = _search_reports(SearchRequest(ticker="META", query="Revenue?"))

        llm_call = get_answer_chain.return_value.invoke.call_args.args[0]
        self.assertIn("No relevant annual-report excerpts", llm_call["context"])
        self.assertEqual(response.sources, [])
        self.assertEqual(
            response.answer, "The reports do not provide enough information."
        )

    def test_blank_ticker_is_rejected(self):
        """Reject blank tickers before querying the document retriever.

        Input:
            A request with a blank ticker value.

        Output:
            A 422 HTTPException is raised and the answer generation pipeline is not
            invoked.
        """
        answer_chain = Mock()
        with patch("api._get_answer_chain", return_value=answer_chain):
            with self.assertRaises(HTTPException) as error:
                _search_reports(SearchRequest(ticker=" ", query="Revenue?"))

        self.assertEqual(error.exception.status_code, 422)
        answer_chain.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
