
"""
ingest_companiesAnnualReport.py
Run this script once to ingest company annual reports into pgvector.
command = "python ingest_companiesAnnualReport.py"
"""

from pathlib import Path
from typing import List, Tuple

from rag_pgVectorIngestion import TABLE_NAME, ingest_CompanyAnnualRpt_docs

SUPPORTED_REPORT_EXTENSIONS = {".docx", ".pdf"}


def find_company_annual_reports(data_dir: Path) -> Tuple[List[str], List[str]]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    company_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if not company_dirs:
        raise FileNotFoundError(f"No company ticker folders found under {data_dir}")

    doc_paths = []
    stock_tickers = []
    for company_dir in company_dirs:
        company_reports = sorted(
            path
            for path in company_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in SUPPORTED_REPORT_EXTENSIONS
        )
        if not company_reports:
            raise FileNotFoundError(
                f"No supported annual report found in ticker folder: {company_dir}. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_REPORT_EXTENSIONS))}"
            )

        ticker = company_dir.name.strip().upper()
        for report_path in company_reports:
            doc_paths.append(str(report_path))
            stock_tickers.append(ticker)

    return doc_paths, stock_tickers


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "Data"
    doc_paths, stock_tickers = find_company_annual_reports(data_dir)
    for report_path, ticker in zip(doc_paths, stock_tickers):
        print(f"Found {report_path} with ticker {ticker}")

    print(f"Ingesting {len(doc_paths)} report(s) into table {TABLE_NAME}...")
    ingest_CompanyAnnualRpt_docs(doc_paths, stock_tickers)
    print(f"Company annual report ingested into table {TABLE_NAME}.")


if __name__ == "__main__":
    main()