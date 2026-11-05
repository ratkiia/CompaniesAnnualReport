import calendar
import hashlib
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
from dotenv import load_dotenv
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pgvector.sqlalchemy import Vector
from pypdf import PdfReader
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    select,
    text,
    tuple_,
)
from sqlalchemy.dialects.postgresql import JSONB

"""
rag_pgVectorIngestion.py

This module is responsible for:
- Connecting to PostgreSQL with pgvector
- Ingesting Annual Reports (PDF, DOCX, TXT) into the database
- Creating a retriever for downstream agents
"""

# Load environment variables from .env file
load_dotenv()

# Connection string for PostgreSQL with pgvector
PG_CONN_STR = os.getenv("PG_CONN_STR")
TABLE_NAME = "companiers_annual_report"
EMBEDDING_DIMENSIONS = 1536

table_metadata = MetaData()
report_table = Table(
    TABLE_NAME,
    table_metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("stock_tickers", String, nullable=False),
    Column("source_file", Text, nullable=False),
    Column("chunk_index", Integer, nullable=False),
    Column("content", Text, nullable=False),
    Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=False),
    Column("creation_date", DateTime(timezone=True)),
    Column("update_date", DateTime(timezone=True)),
    Column("expiration_date", DateTime(timezone=True)),
    Column("version_Id", String, nullable=False),
    Column("metadata", JSONB, nullable=False),
    UniqueConstraint(
        "stock_tickers",
        "source_file",
        "chunk_index",
        name="uq_companiers_annual_report_chunk",
    ),
)
Index("ix_companiers_annual_report_ticker", report_table.c.stock_tickers)

# Text splitter to break long annual report documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # size of each chunk
    chunk_overlap=200, # overlap to preserve context
)


@lru_cache(maxsize=1)
def _get_engine():
    """Create and cache the SQLAlchemy engine used for pgvector queries.

    Input:
        None. Reads the PG_CONN_STR environment value from the project config.

    Output:
        A SQLAlchemy engine configured for PostgreSQL with psycopg support.
    """
    if not PG_CONN_STR:
        raise ValueError("PG_CONN_STR is not set; configure it in the .env file.")
    connection_string = PG_CONN_STR
    if connection_string.startswith("postgresql://"):
        connection_string = connection_string.replace(
            "postgresql://", "postgresql+psycopg://", 1
        )
    elif connection_string.startswith("postgres://"):
        connection_string = connection_string.replace(
            "postgres://", "postgresql+psycopg://", 1
        )
    return create_engine(connection_string)


@lru_cache(maxsize=1)
def _get_embeddings() -> OpenAIEmbeddings:
    """Build the shared embedding client for source-text vectorization.

    Input:
        None.

    Output:
        OpenAIEmbeddings configured for the 1536-dimension text-embedding-3-small
        model used in the project.
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=EMBEDDING_DIMENSIONS,
    )


def _read_document(
    path: str,
) -> Tuple[str, Optional[datetime], Optional[datetime]]:
    """Extract plain text and document metadata from a DOCX or PDF file.

    Input:
        path (str): Absolute or relative path to the support annual-report file.

    Output:
        Tuple[str, Optional[datetime], Optional[datetime]]: The extracted text,
        creation timestamp, and modification timestamp.
    """
    document_path = Path(path)
    if not document_path.is_file():
        raise FileNotFoundError(f"Annual report not found: {document_path}")

    if document_path.suffix.casefold() == ".docx":
        document = DocxDocument(str(document_path))
        text_parts = []
        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                if block.text.strip():
                    text_parts.append(block.text)
            elif isinstance(block, Table):
                for row in block.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip(" |"):
                        text_parts.append(row_text)

        full_text = "\n".join(text_parts)
        created = document.core_properties.created
        modified = document.core_properties.modified
    elif document_path.suffix.casefold() == ".pdf":
        document = PdfReader(str(document_path))
        full_text = "\n".join(page.extract_text() or "" for page in document.pages)
        metadata = document.metadata
        created = metadata.creation_date if metadata else None
        modified = metadata.modification_date if metadata else None
    else:
        raise ValueError(
            f"Unsupported report format {document_path.suffix!r}: {document_path}"
        )

    if not full_text.strip():
        raise ValueError(
            f"No extractable text found in {document_path}. "
            "Scanned image-only PDFs require OCR before ingestion."
        )

    print(f"[DEBUG] Extracted {len(full_text)} characters from {document_path}")
    return full_text, created, modified


def load_docx_text(path: str) -> str:
    """Return the extracted text from a DOCX or PDF file.

    Input:
        path (str): File system path to a supported report file.

    Output:
        str: The report text extracted from the file, without any metadata.
    """
    return _read_document(path)[0]


def _one_year_after(value: Optional[datetime]) -> Optional[datetime]:
    """Calculate the approximate one-year expiration date for a document.

    Input:
        value (Optional[datetime]): The original creation or modification date.

    Output:
        Optional[datetime]: The equivalent timestamp one year later, preserving the
        same month/day when possible.
    """
    if value is None:
        return None

    expiration_year = value.year + 1
    expiration_day = min(value.day, calendar.monthrange(expiration_year, value.month)[1])
    return value.replace(year=expiration_year, day=expiration_day)


def _format_datetime(value: Optional[datetime]) -> Optional[str]:
    """Convert a datetime into ISO format for metadata storage.

    Input:
        value (Optional[datetime]): Timestamp to serialize.

    Output:
        Optional[str]: ISO 8601 string or None when no timestamp exists.
    """
    return value.isoformat() if value is not None else None


def _document_version_id(path: Path) -> str:
    """Create a stable SHA-256 content hash for a source file.

    Input:
        path (Path): File path to the annual report.

    Output:
        str: Hex digest representing the file contents used for version tracking.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_AnnualReport_documents(
    Doc_paths: List[str],
    stock_tickers: List[str],
) -> List[Document]:
    """Create LangChain Document objects from annual-report files.

    Input:
        Doc_paths (List[str]): Report file paths to read.
        stock_tickers (List[str]): Matching ticker values for each report.

    Output:
        List[Document]: Parsed document payloads that include source metadata and
        a content hash for versioning.
    """
    docs: List[Document] = []

    if len(Doc_paths) != len(stock_tickers):
        raise ValueError("Doc_paths and stock_tickers must contain the same number of items.")

    for path, ticker in zip(Doc_paths, stock_tickers):
        if not ticker.strip():
            raise ValueError(f"Stock ticker cannot be empty for report: {path}")

        document_path = Path(path).resolve()
        text, created, modified = _read_document(str(document_path))
        version_id = _document_version_id(document_path)
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "stock_tickers": ticker.strip().upper(),
                    "creation_date": _format_datetime(created),
                    "update_date": _format_datetime(modified),
                    "expiration_date": _format_datetime(_one_year_after(created)),
                    "source_file": str(document_path),
                    "version_Id": version_id,
                },
            )
        )
    return docs


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO timestamp string back into a datetime object.

    Input:
        value (Optional[str]): Serialized ISO timestamp from metadata.

    Output:
        Optional[datetime]: Python datetime or None when no value is stored.
    """
    return datetime.fromisoformat(value) if value is not None else None


def _ensure_report_table(engine) -> None:
    """Create the pgvector table and backfill missing version metadata.

    Input:
        engine: SQLAlchemy engine connected to the PostgreSQL database.

    Output:
        None. The table and required columns are created or migrated in place.
    """
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    table_metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                f'ALTER TABLE "{TABLE_NAME}" '
                'ADD COLUMN IF NOT EXISTS "version_Id" VARCHAR'
            )
        )
        connection.execute(
            text(
                f"""
                WITH legacy_versions AS (
                    SELECT stock_tickers, source_file,
                           md5(string_agg(content, '' ORDER BY chunk_index)) AS version_id
                    FROM "{TABLE_NAME}"
                    WHERE "version_Id" IS NULL
                    GROUP BY stock_tickers, source_file
                )
                UPDATE "{TABLE_NAME}" AS reports
                SET "version_Id" = legacy_versions.version_id
                FROM legacy_versions
                WHERE reports.stock_tickers = legacy_versions.stock_tickers
                  AND reports.source_file = legacy_versions.source_file
                  AND reports."version_Id" IS NULL
                """
            )
        )
        connection.execute(
            text(f'ALTER TABLE "{TABLE_NAME}" ALTER COLUMN "version_Id" SET NOT NULL')
        )


def ingest_CompanyAnnualRpt_docs(
    Doc_paths: List[str],
    stock_tickers: List[str],
) -> None:
    """Ingest annual-report files into PostgreSQL and store vector embeddings.

    Input:
        Doc_paths (List[str]): Report file paths to read and ingest.
        stock_tickers (List[str]): Matching ticker symbols for each path.

    Output:
        None. New or changed reports are chunked, embedded, and persisted to the
        configured pgvector table.
    """
    # 1. Build documents and hash each source file.
    docs = build_AnnualReport_documents(Doc_paths, stock_tickers)
    print(f"[DEBUG] Loaded {len(docs)} documents")

    if not docs:
        raise ValueError("No annual report files were provided for ingestion.")

    engine = _get_engine()
    _ensure_report_table(engine)

    document_versions = {
        (doc.metadata["stock_tickers"], doc.metadata["source_file"]): doc.metadata[
            "version_Id"
        ]
        for doc in docs
    }
    existing_versions = {}
    with engine.connect() as connection:
        existing_rows = connection.execute(
            select(
                report_table.c.stock_tickers,
                report_table.c.source_file,
                report_table.c.version_Id,
            ).where(
                tuple_(
                    report_table.c.stock_tickers,
                    report_table.c.source_file,
                ).in_(list(document_versions))
            )
        )
        for ticker, source_file, version_id in existing_rows:
            existing_versions[(ticker, source_file)] = version_id

    changed_docs = [
        doc
        for doc in docs
        if existing_versions.get(
            (doc.metadata["stock_tickers"], doc.metadata["source_file"])
        )
        != doc.metadata["version_Id"]
    ]
    skipped_count = len(docs) - len(changed_docs)
    if skipped_count:
        print(f"[INFO] Skipped {skipped_count} unchanged report(s)")
    if not changed_docs:
        print("[INFO] No new or changed reports; embeddings are up to date.")
        return

    # 2. Split only new or changed reports.
    chunks = splitter.split_documents(changed_docs)
    print(f"[DEBUG] Created {len(chunks)} chunks")

    if not chunks:
        raise ValueError("No text chunks were extracted from the changed reports.")

    # 3. Generate embeddings only for chunks that need to be inserted.
    vectors = _get_embeddings().embed_documents(
        [chunk.page_content for chunk in chunks]
    )
    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Expected {len(chunks)} embeddings, received {len(vectors)}."
        )

    rows = []
    chunk_indexes = {}
    source_tickers = set()
    for chunk, vector in zip(chunks, vectors):
        metadata = chunk.metadata
        ticker = metadata["stock_tickers"]
        source_file = metadata["source_file"]
        source_tickers.add((ticker, source_file))

        chunk_key = (ticker, source_file)
        chunk_index = chunk_indexes.get(chunk_key, 0)
        chunk_indexes[chunk_key] = chunk_index + 1
        rows.append(
            {
                "stock_tickers": ticker,
                "source_file": source_file,
                "chunk_index": chunk_index,
                "content": chunk.page_content,
                "embedding": vector,
                "creation_date": _parse_datetime(metadata["creation_date"]),
                "update_date": _parse_datetime(metadata["update_date"]),
                "expiration_date": _parse_datetime(metadata["expiration_date"]),
                "version_Id": metadata["version_Id"],
                "metadata": {**metadata, "chunk_index": chunk_index},
            }
        )

    with engine.begin() as connection:
        for ticker, source_file in source_tickers:
            connection.execute(
                delete(report_table).where(
                    report_table.c.stock_tickers == ticker,
                    report_table.c.source_file == source_file,
                )
            )
        connection.execute(report_table.insert(), rows)

    print(f"[DEBUG] Inserted {len(rows)} chunks into PostgreSQL table {TABLE_NAME}")


def get_company_annual_report_version_id(
    stock_ticker: str,
    source_file: str,
) -> str:
    """Fetch the stored version hash for a specific ticker and report file.

    Input:
        stock_ticker (str): Company ticker to query.
        source_file (str): Path to the report file on disk.

    Output:
        str: The document version identifier stored in the database.
    """
    ticker = stock_ticker.strip().upper()
    if not ticker:
        raise ValueError("stock_ticker cannot be empty.")

    statement = (
        select(report_table.c.version_Id)
        .where(
            report_table.c.stock_tickers == ticker,
            report_table.c.source_file == str(Path(source_file).resolve()),
        )
        .distinct()
        .limit(2)
    )
    with _get_engine().connect() as connection:
        version_ids = [row[0] for row in connection.execute(statement)]

    if not version_ids:
        raise LookupError(
            f"No ingested report found for ticker {ticker} and source {source_file}."
        )
    if len(version_ids) > 1:
        raise RuntimeError(
            f"Multiple document versions found for ticker {ticker} and source {source_file}."
        )
    if version_ids[0] is None:
        raise RuntimeError(
            "This report has no version_Id yet; re-ingest it to enable versioned caching."
        )
    return version_ids[0]


def build_company_annual_report_cache_key(
    stock_ticker: str,
    source_file: str,
    query: str,
) -> str:
    """Create a versioned cache key for a ticker, report, and user query.

    Input:
        stock_ticker (str): Company ticker to access.
        source_file (str): Path to the annual report file.
        query (str): User query for the cached result.

    Output:
        str: A deterministic cache identifier that changes whenever the report or
        query changes.
    """
    version_id = get_company_annual_report_version_id(stock_ticker, source_file)
    source_id = hashlib.sha256(
        str(Path(source_file).resolve()).encode("utf-8")
    ).hexdigest()
    query_id = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return (
        f"annual-report:{stock_ticker.strip().upper()}:"
        f"{source_id}:{version_id}:{query_id}"
    )


class CompanyAnnualReportRetriever(BaseRetriever):
    stock_ticker: str
    k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Return relevant report chunks for a search query using vector similarity.

        Input:
            query (str): Natural-language search string.
            run_manager: Callback manager used by LangChain when the retriever runs.

        Output:
            List[Document]: The top matching chunk documents for the configured
            ticker, including their metadata.
        """
        query_vector = _get_embeddings().embed_query(query)
        distance = report_table.c.embedding.cosine_distance(query_vector)
        statement = (
            select(report_table.c.content, report_table.c["metadata"])
            .where(report_table.c.stock_tickers == self.stock_ticker)
            .order_by(distance)
            .limit(self.k)
        )

        with _get_engine().connect() as connection:
            results = connection.execute(statement).mappings().all()

        return [
            Document(page_content=row["content"], metadata=row["metadata"])
            for row in results
        ]


def get_company_annual_report_retriever(stock_ticker: str) -> CompanyAnnualReportRetriever:
    """Create a retriever that searches a single ticker's annual reports.

    Input:
        stock_ticker (str): Desired company ticker to filter on.

    Output:
        CompanyAnnualReportRetriever: A retriever instance that supports
        .invoke(query) and returns the most relevant chunks.
    """
    if not stock_ticker.strip():
        raise ValueError("stock_ticker cannot be empty.")
    return CompanyAnnualReportRetriever(stock_ticker=stock_ticker.strip().upper())