#!/usr/bin/env python3
"""
One-time sanity check after ingest.py: document count + similarity_search smoke test.

Run: python verify_chroma.py
"""

import os

from dotenv import load_dotenv

from rag_common import get_embeddings, get_vectorstore

load_dotenv()


def main() -> None:
    expected = int(os.getenv("EXPECTED_DOC_COUNT", "141000"))
    test_query = os.getenv(
        "VERIFY_TEST_QUERY",
        "Tata Consultancy Services London",
    )

    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)
    count = vectorstore._collection.count()

    print(f"Chroma collection document count: {count}")
    if count == expected:
        print(f"OK: matches EXPECTED_DOC_COUNT ({expected}).")
    else:
        print(
            f"WARNING: expected {expected} documents (set EXPECTED_DOC_COUNT if different)."
        )

    print(f"\nTest similarity_search (k=5): {test_query!r}\n")
    docs = vectorstore.similarity_search(test_query, k=5)
    if not docs:
        print("No results returned — collection may be empty or embedding mismatch.")
        return

    for i, doc in enumerate(docs):
        meta = doc.metadata
        preview = doc.page_content.replace("\n", " ")[:240]
        print(f"[{i}] sheet={meta.get('sheet')} row={meta.get('row_index')}")
        print(f"    {preview}...")
        print()


if __name__ == "__main__":
    main()
