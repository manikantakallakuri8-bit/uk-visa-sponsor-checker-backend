"""
Shared RAG helpers: local embeddings, Chroma connection, spreadsheet → Documents.
Used by main.py (API), ingest.py (bulk load), and verify_chroma.py (health check).
"""

import os
import re
from typing import List

import pandas as pd
from rapidfuzz import fuzz, process
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()


def normalise_company_name(name: str) -> str:
    name = name.lower().strip()
    number_map = {
        r"\bone\b": "1",
        r"\btwo\b": "2",
        r"\bthree\b": "3",
        r"\bfour\b": "4",
        r"\bfive\b": "5",
        r"\bsix\b": "6",
        r"\bseven\b": "7",
        r"\beight\b": "8",
        r"\bnine\b": "9",
    }
    for word, digit in number_map.items():
        name = re.sub(word, digit, name)
    name = re.sub(r"\bltd\b", "limited", name)
    name = re.sub(r"\bco\b", "company", name)
    name = re.sub(r"%", " percent ", name)
    name = re.sub(r"&", " and ", name)
    name = re.sub(r"@", " at ", name)
    name = re.sub(r"\+", " plus ", name)
    name = re.sub(r"£", " pounds ", name)
    name = re.sub(r"#", " number ", name)
    name = re.sub(r"\bno\.?\s*(\d+)", r"number \1", name)
    name = re.sub(r"[-.]", " ", name)
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def fuzzy_fallback(query: str, csv_path: str = "tier2.csv", top_n: int = 5):
    if not os.path.isfile(csv_path):
        return []
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
    except Exception:
        return []

    # handle column name variations safely
    name_col = next((c for c in df.columns if "organisation" in c.lower()), None)
    town_col = next((c for c in df.columns if "town" in c.lower()), None)
    route_col = next((c for c in df.columns if "route" in c.lower()), None)
    rating_col = next(
        (c for c in df.columns if "rating" in c.lower() or "type" in c.lower()),
        None,
    )

    if not name_col:
        return []

    df[name_col] = df[name_col].str.strip()
    df[name_col] = df[name_col].apply(normalise_company_name)
    names = df[name_col].tolist()

    results = process.extract(
        normalise_company_name(query),
        names,
        scorer=fuzz.token_sort_ratio,
        limit=top_n,
    )
    matches = []
    for name, score, idx in results:
        if score > 50:
            row = df.iloc[idx]
            matches.append(
                {
                    "organisation_name": name,
                    "town": str(row[town_col]).strip() if town_col else "",
                    "route": str(row[route_col]).strip() if route_col else "",
                    "rating": str(row[rating_col]).strip() if rating_col else "",
                    "fuzzy_score": score,
                }
            )
    return matches


def _cell_text(val) -> str:
    """Strip leading spaces from cell values (register CSVs often pad with spaces)."""
    return str(val).lstrip()


def _norm_col_name(col) -> str:
    s = str(col).strip().lower().replace(" ", "_")
    return s.replace("&", "and").replace("/", "_")


def _cell_by_normalized_keys(row, columns, *candidate_keys: str) -> str:
    """Return first non-empty cell whose column normalizes to one of candidate_keys."""
    norm_to_orig = {_norm_col_name(c): c for c in columns}
    for key in candidate_keys:
        orig = norm_to_orig.get(key)
        if orig is not None:
            val = _cell_text(row[orig]).strip()
            if val:
                return val
    return ""


def _organisation_name_from_row(row, columns) -> str:
    """Best-effort org name for metadata / context (sponsor register column names)."""
    name = _cell_by_normalized_keys(
        row,
        columns,
        "organisation_name",
        "organization_name",
    )
    return name if name else "unknown"


def _register_row_metadata(row, columns) -> dict:
    """Metadata fields for UK sponsor register rows (column names vary slightly)."""
    org = _organisation_name_from_row(row, columns)
    town = _cell_by_normalized_keys(
        row, columns, "town", "town_city", "city"
    )
    route = _cell_by_normalized_keys(row, columns, "route")
    type_rating = _cell_by_normalized_keys(
        row,
        columns,
        "type_and_rating",
        "type_rating",
        "type",
        "rating",
    )
    return {
        "organisation_name": org,
        "town": town if town else "Unknown",
        "route": route if route else "Unknown",
        "type_rating": type_rating if type_rating else "Unknown",
    }


def _row_page_content(row, columns) -> str:
    return " | ".join([f"{col}: {_cell_text(row[col])}" for col in columns])


def get_embeddings() -> HuggingFaceEmbeddings:
    model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model)


def get_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "sheet_rag_collection")
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


def spreadsheet_to_documents(path: str, filename: str) -> List[Document]:
    """One row = one document. Used by FastAPI upload ingest."""
    docs: List[Document] = []
    lower_name = filename.lower()

    if lower_name.endswith(".csv"):
        df = pd.read_csv(path)
        for idx, row in df.fillna("").iterrows():
            row_text = _row_page_content(row, df.columns)
            meta = _register_row_metadata(row, df.columns)
            meta.update(
                {
                    "source_type": "csv",
                    "sheet": "csv",
                    "row_index": int(idx),
                }
            )
            docs.append(Document(page_content=row_text, metadata=meta))
        return docs

    workbook = pd.read_excel(path, sheet_name=None)
    for sheet_name, df in workbook.items():
        for idx, row in df.fillna("").iterrows():
            row_text = _row_page_content(row, df.columns)
            meta = _register_row_metadata(row, df.columns)
            meta.update(
                {
                    "source_type": "xlsx",
                    "sheet": sheet_name,
                    "row_index": int(idx),
                }
            )
            docs.append(Document(page_content=row_text, metadata=meta))
    return docs


def csv_chunk_to_documents(
    chunk: pd.DataFrame,
    row_index_base: int,
    *,
    source_type: str = "csv",
    sheet: str = "csv",
) -> List[Document]:
    """
    Build documents from a pandas CSV chunk.
    row_index_base makes row_index global across chunks (chunked read_csv resets index each time).
    """
    docs: List[Document] = []
    chunk_filled = chunk.fillna("")
    for i in range(len(chunk_filled)):
        row = chunk_filled.iloc[i]
        idx = row_index_base + i
        row_text = _row_page_content(row, chunk.columns)
        meta = _register_row_metadata(row, chunk.columns)
        meta.update(
            {
                "source_type": source_type,
                "sheet": sheet,
                "row_index": int(idx),
            }
        )
        docs.append(Document(page_content=row_text, metadata=meta))
    return docs


def search_sponsor_register(query: str, vectorstore) -> str:
    import re
    stripped = re.sub(
        r"(?i)(does|do|can|is|will|would|has|have|"
        r"sponsor|skilled worker|visa|visas|licence|license|"
        r"uk|the|a|an|for|me|us|them|they|it|"
        r"i have an interview at|i found a job at|"
        r"next week)\b",
        " ",
        query,
    )
    stripped = re.sub(r"\s+", " ", stripped).strip()
    clean = normalise_company_name(stripped if stripped else query)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 24},
    )
    docs = retriever.invoke(clean)
    fuzzy = fuzzy_fallback(clean)

    results = []
    for doc in docs:
        name = doc.metadata.get("organisation_name", "").strip()
        town = doc.metadata.get("town", "").strip()
        route = doc.metadata.get("route", "").strip()
        rating = doc.metadata.get("type_rating", "").strip()
        if name:
            results.append(f"{name} | {town} | {route} | {rating}")
        # print("----chroma results----")
        # print(results)

    for m in fuzzy:
        results.append(
            f"{m['organisation_name']} | {m['town']} | "
            f"{m['route']} | {m['rating']} (fuzzy {m['fuzzy_score']}%)"
        )
        # print("----fuzzy results----")
        # print(results)

    if not results:
        return "No matching sponsor found in the UK Home Office register."
    return "Register matches:\n" + "\n".join(results[:8])
