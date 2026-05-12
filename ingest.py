#!/usr/bin/env python3
"""
One-time bulk ingest: reads tier2.csv from disk (no FastAPI), embeds rows with
sentence-transformers, writes to ./chroma_db. Safe for large CSVs via chunked reads.

Run: python ingest.py
Requires: tier2.csv in project root or set TIER2_CSV_PATH in .env
"""

import os

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from rag_common import csv_chunk_to_documents, get_embeddings, get_vectorstore

load_dotenv()


def count_csv_data_rows(path: str) -> int:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return max(0, sum(1 for _ in f) - 1)


def main() -> None:
    csv_path = os.getenv("TIER2_CSV_PATH", "tier2.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(
            f"CSV not found: {csv_path}. Place tier2.csv here or set TIER2_CSV_PATH."
        )

    chunk_rows = int(os.getenv("INGEST_CHUNK_ROWS", "2000"))
    add_batch = int(os.getenv("INGEST_ADD_BATCH", "500"))

    total_rows = count_csv_data_rows(csv_path)
    print(f"Ingesting {csv_path} (~{total_rows} data rows), chunk={chunk_rows}, add_batch={add_batch}")

    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)
    vectorstore.delete_collection()
    vectorstore = get_vectorstore(embeddings)

    row_base = 0
    with tqdm(total=total_rows, unit="rows", desc="Chroma ingest") as pbar:
        for chunk in pd.read_csv(csv_path, chunksize=chunk_rows):
            docs = csv_chunk_to_documents(chunk, row_base)
            for i in range(0, len(docs), add_batch):
                vectorstore.add_documents(docs[i : i + add_batch])
            row_base += len(chunk)
            pbar.update(len(chunk))

    final_count = vectorstore._collection.count()
    print(f"Done. Rows processed: {row_base}. Collection count: {final_count}")


if __name__ == "__main__":
    main()
