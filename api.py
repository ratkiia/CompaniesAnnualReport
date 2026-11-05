import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from rag_pgVectorIngestion import get_company_annual_report_retriever

load_dotenv()

app = FastAPI(title="Annual Report RAG API")

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only the supplied annual-report excerpts. "
            "Do not use outside knowledge. If the excerpts do not contain the "
            "answer, say that the reports do not provide enough information. "
            "When making claims, cite the matching excerpt with its [number].",
        ),
        ("human", "Question: {question}\n\nAnnual-report excerpts:\n{context}"),
    ]
)


class SearchRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=20)
    query: str = Field(min_length=1, max_length=2000)


class SearchSource(BaseModel):
    number: int
    file_name: str
    chunk_index: Optional[int] = None
    excerpt: str


class SearchResponse(BaseModel):
    ticker: str
    answer: str
    sources: List[SearchSource]


@lru_cache(maxsize=1)
def _get_answer_chain():
    """Build and cache the LLM answer-generation pipeline.

    Input:
        None. Reads the configured chat model from the environment variable
        OPENAI_CHAT_MODEL, defaulting to gpt-4o-mini when unset.

    Output:
        A LangChain runnable chain that accepts a question and context string,
        then returns a grounded answer generated from the supplied excerpts.
    """
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    return answer_prompt | llm | StrOutputParser()


def _search_reports(request: SearchRequest) -> SearchResponse:
    """Retrieve relevant report excerpts and synthesize a grounded answer.

    Input:
        request (SearchRequest): A request containing the company ticker and the
        user question to answer.

    Output:
        SearchResponse: A normalized response containing the ticker, the answer,
        and numbered source excerpts with file names and chunk metadata.

    Raises:
        HTTPException: Raised when the ticker or query is blank so the API returns
        a 422 validation error.
    """
    ticker = request.ticker.strip().upper()
    query = request.query.strip()
    if not ticker:
        raise HTTPException(status_code=422, detail="Ticker cannot be blank.")
    if not query:
        raise HTTPException(status_code=422, detail="Query cannot be blank.")

    documents = get_company_annual_report_retriever(ticker).invoke(query)
    sources = [
        SearchSource(
            number=index,
            file_name=Path(
                str(document.metadata.get("source_file", "Annual report"))
            ).name,
            chunk_index=document.metadata.get("chunk_index"),
            excerpt=document.page_content,
        )
        for index, document in enumerate(documents, start=1)
    ]
    context = "\n\n".join(
        f"[{source.number}] {source.file_name} "
        f"(chunk {source.chunk_index})\n{source.excerpt}"
        for source in sources
    )
    if not context:
        context = "No relevant annual-report excerpts were retrieved."

    answer = _get_answer_chain().invoke({"question": query, "context": context})
    return SearchResponse(ticker=ticker, answer=answer, sources=sources)


@app.post("/search", response_model=SearchResponse)
def search_reports(request: SearchRequest) -> SearchResponse:
    """FastAPI endpoint that answers a company-question search request.

    Input:
        request (SearchRequest): The HTTP body payload with a ticker and query.

    Output:
        SearchResponse: The generated answer and source evidence for the query.
    """
    return _search_reports(request)
