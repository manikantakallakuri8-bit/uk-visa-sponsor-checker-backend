# uk-visa-sponsor-checker
RAG-powered API to check if a UK company holds a Skilled Worker sponsorship licence - FastAPI + ChromaDB + sentence-transformers + Groq + Angular frontend

# UK Visa Sponsor Checker

A full-stack RAG (Retrieval-Augmented Generation) application that lets job seekers instantly check whether a UK company holds a valid Skilled Worker sponsorship licence - without manually searching the 141,000-row Home Office register.

# The Problem
International job seekers in the UK waste hours cross-checking companies from job boards against the government sponsor register. Company names on job posts rarely match registered names exactly - abbreviations, symbols, hyphens, and initials cause manual searches to fail silently.

# The Solution
A three-layer search pipeline:
1. **Query normalisation** — handles symbols (%, &, +), abbreviations (Ltd → Limited), and number variants (1-2 → 1 to 2) before searching
2. **Semantic search** — sentence-transformers embeddings + ChromaDB retrieve the most contextually similar sponsor records
3. **Fuzzy fallback** — RapidFuzz token matching catches character-level variations the embedding model misses

## Features
- Natural language queries - "Does Tata Consultancy sponsor Skilled Workers?"
- Handles fuzzy company names - typos, abbreviations, symbols
- Structured JSON response - sponsored status, rating, town, visa route
- Bulk check endpoint - verify multiple companies in one call
- Angular frontend with real-time search
- Zero cost to run - free embeddings, free Groq tier, free ChromaDB
Results are passed to Groq's llama-3.1-8b-instant model which generates a clear, human-readable answer with sponsorship status and location.

# Spreadsheet RAG App (LangChain + ChromaDB + Groq + FastAPI)

This API ingests `.xlsx` or `.csv` files, stores local sentence-transformer embeddings in ChromaDB, retrieves relevant rows, and asks a Groq-hosted LLM to generate the final answer.

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Edit `.env` and set:

- `GROQ_API_KEY`
- optionally `GROQ_BASE_URL`, `GROQ_MODEL`, `EMBEDDING_MODEL`

## 2) Run FastAPI

```bash
uvicorn main:app --reload
```

## 3) API Endpoints

- `GET /health` - health check
- `POST /ingest-xlsx` - upload an `.xlsx` or `.csv` file and index rows in Chroma
- `POST /ask` - ask a question against indexed spreadsheet content

Notes:
- Each spreadsheet row is stored as one document (no text splitting).
- By default, ingestion resets the collection each time (`RESET_COLLECTION_ON_INGEST=true`).

### Ingest Example

```bash
curl -X POST "http://127.0.0.1:8000/ingest-xlsx" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@sample.xlsx" 
```

### Ask Example

```bash
curl -X POST "http://127.0.0.1:8000/ask" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"Does xyz company has uk sponsorship licence?\",\"top_k\":8}"
```
