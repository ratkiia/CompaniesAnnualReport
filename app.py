"""Streamlit UI for querying ingested annual-report data via the FastAPI backend."""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Annual Report Search")
st.title("Annual report search")
st.caption("Ask a question about an ingested company's annual reports.")

with st.form("report_search"):
    ticker = st.text_input("Company ticker", placeholder="MSFT")
    query = st.text_area(
        "Your question",
        placeholder="What were the company's major risks?",
        height=100,
    )
    submitted = st.form_submit_button("Search")

if submitted:
    if not ticker.strip() or not query.strip():
        st.error("Enter both a company ticker and a question.")
    else:
        try:
            response = requests.post(
                f"{API_URL}/search",
                json={"ticker": ticker, "query": query},
                timeout=120,
            )
        except requests.RequestException as error:
            st.error(f"Could not reach the RAG API: {error}")
        else:
            if not response.ok:
                st.error(
                    f"The RAG API returned HTTP {response.status_code}: "
                    f"{response.text}"
                )
            else:
                result = response.json()
                st.subheader("Answer")
                st.write(result["answer"])

                if result["sources"]:
                    st.subheader("Sources")
                    for source in result["sources"]:
                        chunk = source["chunk_index"]
                        title = f'[{source["number"]}] {source["file_name"]}'
                        if chunk is not None:
                            title += f" - chunk {chunk}"
                        with st.expander(title):
                            st.write(source["excerpt"])
                else:
                    st.info("No matching report excerpts were found.")
