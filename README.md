# Companies Annual Report RAG

This project ingests company annual reports into a PostgreSQL database with
pgvector, then retrieves relevant text chunks by stock ticker and semantic
similarity.

## Requirements

- Python 3.10 or later
- PostgreSQL with the pgvector extension available
- An OpenAI API key for the `text-embedding-3-small` embedding model
- `python-docx` and `pypdf` (included in `requirements.txt`)

## Setup

From the project directory, create and activate a virtual environment and
install the project dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file in the project directory:

```dotenv
OPENAI_API_KEY=your-openai-api-key
PG_CONN_STR=postgresql://username:password@localhost:5432/vectordb
```

Use your database's actual credentials and connection address. Keep `.env`
private; it is excluded by `.gitignore`.

The database user needs permission to connect to the database, create the
`vector` extension if it is not already enabled, and create/alter the report
table. If the user cannot create extensions, have a database administrator
enable pgvector first:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Organize report files

Create one folder per stock ticker under `Data`. The folder name is used as the
ticker value in the database. Reports can have any filename and can be nested
in subfolders.

```text
Data/
  META/
    Meta-Annual-Report.pdf
  MSFT/
    reports/
      2026_AnnualReport.docx
```

The supported formats are:

- `.docx`
- `.pdf`

The extension check is case-insensitive. Legacy `.doc` files and `.docs` files
are not supported. Each ticker folder must contain at least one supported file.
Scanned/image-only PDFs need OCR before ingestion; the current parser requires
extractable PDF text and reports an error if it finds none.

## Ingest reports

Run the ingestion script from the project directory:

```powershell
python .\ingest_companiesAnnualReport.py
```

For each supported file, the script:

1. Infers the ticker from its parent company folder.
2. Extracts text and embedded creation/modification dates from the DOCX or PDF.
3. Computes a SHA-256 `version_Id` from the source file's bytes.
4. Checks the stored version for that ticker and file path.
5. Skips unchanged files. New or changed files are split into 1,000-character
   chunks with 200 characters of overlap, embedded, and stored in PostgreSQL.
6. Replaces the previous chunks for a changed file.

The first ingestion creates the pgvector extension and table if needed. The
script also adds/backfills `version_Id` for an existing table. Ingestion makes
OpenAI API calls for reports that are new or have changed.

## Database table and metadata

The physical table name is `companiers_annual_report`. It contains one row per
text chunk, including:

| Column | Description |
| --- | --- |
| `id` | Row identifier |
| `stock_tickers` | Uppercase ticker inferred from the folder name |
| `source_file` | Absolute source-file path |
| `chunk_index` | Zero-based chunk position within the report |
| `content` | Text for the chunk |
| `embedding` | 1,536-dimensional `text-embedding-3-small` vector |
| `creation_date` | Creation date from DOCX/PDF metadata, if present |
| `update_date` | Modification date from DOCX/PDF metadata, if present |
| `expiration_date` | Creation date plus one year, if the creation date is present |
| `version_Id` | SHA-256 content version for the source file |
| `metadata` | JSONB copy of document metadata and chunk index |

Dates are stored as timezone-aware timestamps when the source provides them;
otherwise their database values are `NULL`. `expiration_date` is metadata only:
the application does not automatically delete expired rows.

Example query to inspect ingested reports:

```sql
SELECT
    stock_tickers,
    source_file,
    "version_Id",
    creation_date,
    update_date,
    expiration_date,
    COUNT(*) AS chunk_count
FROM companiers_annual_report
GROUP BY
    stock_tickers,
    source_file,
    "version_Id",
    creation_date,
    update_date,
    expiration_date
ORDER BY stock_tickers, source_file;
```

## Retrieve report chunks

`get_company_annual_report_retriever` returns a LangChain retriever that
searches the table by cosine similarity and filters results by ticker:

```python
from rag_pgVectorIngestion import get_company_annual_report_retriever

retriever = get_company_annual_report_retriever("MSFT")
documents = retriever.invoke("What were the company's major risks?")
```

The retriever returns up to five chunks by default. Each returned LangChain
`Document` includes the stored metadata.

## Search reports with the web UI

The FastAPI service retrieves up to five relevant report chunks, sends those
excerpts to an OpenAI chat model, and returns a grounded answer with numbered
source excerpts. The model is instructed to use only retrieved report content.
Set `OPENAI_CHAT_MODEL` in `.env` to choose another supported chat model; the
default is `gpt-4o-mini`. The API uses the existing `OPENAI_API_KEY`.

Install dependencies after updating `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

Start the API from the project directory:

```powershell
python -m uvicorn api:app --reload
```

In a second terminal, start the Streamlit UI:

```powershell
streamlit run app.py
```

Enter an ingested company ticker and a question. The answer appears above the
expandable excerpts used as its sources. To run the UI against an API at a
different address, set `RAG_API_URL` (for example,
`RAG_API_URL=http://localhost:8000`) in the UI process environment.

The API is also available directly at `POST /search`:

```json
{
  "ticker": "MSFT",
  "query": "What were the company's major risks?"
}
```

Its response includes an `answer` and a `sources` array with report filenames,
chunk indexes, and excerpts. Keep the API private or add authentication before
exposing it beyond a trusted local environment.

## API function reference

The service exposes a small set of Python functions that power the API flow:

- `_get_answer_chain()`
  - Input: no explicit arguments; it reads `OPENAI_CHAT_MODEL` from the process
    environment.
  - Output: a cached LangChain prompt + LLM + parser chain that turns a question
    and context into a final answer string.

- `_search_reports(request: SearchRequest) -> SearchResponse`
  - Input: a `SearchRequest` payload containing a ticker and a query string.
  - Output: a `SearchResponse` with the normalized ticker, final answer, and
    numbered source excerpts from the matching annual reports.

- `search_reports(request: SearchRequest) -> SearchResponse`
  - Input: the FastAPI request object from `POST /search`.
  - Output: the same `SearchResponse` generated by `_search_reports`, returned to
    the API client.

This keeps the retrieval, validation, and answer-generation logic separated while
making the API endpoint itself simple and reusable.

## Versioned KV-cache keys

Use `build_company_annual_report_cache_key` to create a cache key for a
specific ticker, source file, and query:

```python
from rag_pgVectorIngestion import build_company_annual_report_cache_key

cache_key = build_company_annual_report_cache_key(
    "MSFT",
    r"Data\MSFT\2026_AnnualReport.docx",
    "What were the company's major risks?",
)
```

The key includes the stored `version_Id`; after a changed report is ingested,
the new version produces a different key, so the old cached value is no longer
selected by that key. The helper builds a key but does not implement a KV
store, read/write cache values, or delete old cache entries.
