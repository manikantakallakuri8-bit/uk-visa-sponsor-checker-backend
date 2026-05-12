# Spreadsheet RAG App (LangChain + ChromaDB + Groq + FastAPI)

This API ingests `.xlsx` or `.csv` files, stores local sentence-transformer embeddings in ChromaDB, retrieves relevant rows, and asks a Groq-hosted LLM to generate the final answer.

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your key:

```bash
copy .env.example .env
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
  -d "{\"question\":\"What is the total sales for Q1?\",\"top_k\":4}"
```
