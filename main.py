"""
Running API: connects to ./chroma_db on startup (does not preload tier2.csv).
Ingest via POST /ingest-xlsx; questions via POST /ask (Groq).
"""

import os
import tempfile
from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from rag_common import (
    fuzzy_fallback,
    get_embeddings,
    get_vectorstore,
    normalise_company_name,
    spreadsheet_to_documents,
)

load_dotenv()


def get_llm() -> ChatOpenAI:
    api_key = os.getenv("GROQ_API_KEY")
    base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is missing.")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.embeddings = get_embeddings()
    app.state.llm = get_llm()
    app.state.vectorstore = get_vectorstore(app.state.embeddings)
    yield


app = FastAPI(title="Spreadsheet RAG with Chroma + Groq", lifespan=lifespan)

# Angular (or any browser client) on another origin triggers a CORS preflight (OPTIONS).
# Without CORSMiddleware, OPTIONS /ask returns 405 Method Not Allowed.
_default_cors = "http://localhost:4200,http://127.0.0.1:4200,http://localhost:3000"
_cors_origins = [
    o.strip() for o in os.getenv("CORS_ORIGINS", _default_cors).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    top_k: int = 8


class SponsorMatch(BaseModel):
    organisation_name: str
    town: str
    route: str
    rating: str
    row_index: int


class SponsorResponse(BaseModel):
    query: str
    sponsored: bool
    status: str  # "Licensed" | "Not Found" | "Uncertain"
    summary: str
    matches: List[SponsorMatch]
    total_matches_found: int
    data_source: str = "UK Home Office Register of Licensed Sponsors"
    disclaimer: str = (
        "Always verify at gov.uk/government/publications/register-of-licensed-sponsors-workers"
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest-xlsx")
async def ingest_spreadsheet(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")
    if not file.filename.lower().endswith((".xlsx", ".csv")):
        raise HTTPException(status_code=400, detail="Only .xlsx and .csv files are supported.")

    try:
        suffix = ".csv" if file.filename.lower().endswith(".csv") else ".xlsx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        raw_docs = spreadsheet_to_documents(temp_path, file.filename)
        if not raw_docs:
            raise HTTPException(status_code=400, detail="No rows found in input file.")

        vectorstore: Chroma = app.state.vectorstore
        reset_on_ingest = os.getenv("RESET_COLLECTION_ON_INGEST", "true").lower() == "true"

        if reset_on_ingest:
            vectorstore.delete_collection()
            app.state.vectorstore = get_vectorstore(app.state.embeddings)
            vectorstore = app.state.vectorstore
            previous_count = 0
        else:
            previous_count = vectorstore._collection.count()

        # Each row is a single document; no text splitting.
        vectorstore.add_documents(raw_docs)
        return {
            "message": "Spreadsheet ingested successfully.",
            "reset_collection": reset_on_ingest,
            "existing_documents_before_ingest": previous_count,
            "documents": len(raw_docs),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    finally:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/ask", response_model=SponsorResponse)
def ask_question(payload: AskRequest) -> SponsorResponse:
    # import pdb; pdb.set_trace()
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    def _row_idx(meta: dict) -> int:
        raw: Optional[object] = meta.get("row_index", -1)
        try:
            return int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return -1

    try:
        # Step 1 — normalise query
        clean_query = normalise_company_name(payload.question)

        # Step 2 — semantic search
        vectorstore: Chroma = app.state.vectorstore
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": payload.top_k, "fetch_k": payload.top_k * 4},
        )
        semantic_docs = retriever.invoke(clean_query)

        # Step 3 — fuzzy fallback
        fuzzy_results = fuzzy_fallback(
            clean_query, os.getenv("TIER2_CSV_PATH", "tier2.csv")
        )

        # Step 4 — build context from semantic results
        semantic_context = "\n\n".join(
            [
                f"Company: {doc.metadata.get('organisation_name', 'Unknown')} | "
                f"Town: {doc.metadata.get('town', 'Unknown')} | "
                f"Route: {doc.metadata.get('route', 'Unknown')} | "
                f"Rating: {doc.metadata.get('type_rating', 'Unknown')}"
                for doc in semantic_docs
            ]
        )

        # Step 5 — build context from fuzzy results
        fuzzy_context = "\n".join(
            [
                f"Company: {m['organisation_name']} | "
                f"Town: {m['town']} | "
                f"Route: {m['route']} | "
                f"Rating: {m['rating']} | "
                f"Match confidence: {m['fuzzy_score']}%"
                for m in fuzzy_results
            ]
        )

        # Step 6 — merge both contexts
        full_context = ""
        if semantic_context:
            full_context += f"Semantic matches:\n{semantic_context}\n\n"
        if fuzzy_context:
            full_context += f"Fuzzy matches:\n{fuzzy_context}"

        if not full_context:
            return SponsorResponse(
                query=payload.question,
                sponsored=False,
                status="Not Found",
                summary="No matching company found in the UK sponsor register.",
                matches=[],
                total_matches_found=0,
            )

        # Step 7 — LLM prompt
        prompt = (
            "You are a UK visa sponsorship checker assistant.\n"
            "Based ONLY on the register data below, answer in ONE sentence:\n"
            "- Does the company hold a Skilled Worker sponsor licence?\n"
            "- If yes: state the company name, town, and rating.\n"
            "- If no exact match: say 'Not found in the register.'\n\n"
            f"Company searched: {payload.question}\n"
            f"Normalised search term: {clean_query}\n\n"
            f"Register data:\n{full_context}\n\n"
            "One sentence answer:"
        )

        response = app.state.llm.invoke(prompt)

        # Step 8 — build matches list
        answer_lower = response.content.lower()
        sponsored = (
            "not found" not in answer_lower and "do not know" not in answer_lower
        )

        all_matches: List[SponsorMatch] = []

        # from semantic docs
        for doc in semantic_docs:
            name = doc.metadata.get("organisation_name", "").strip()
            if name:
                all_matches.append(
                    SponsorMatch(
                        organisation_name=name,
                        town=str(doc.metadata.get("town", "")).strip(),
                        route=str(doc.metadata.get("route", "")).strip(),
                        rating=str(doc.metadata.get("type_rating", "")).strip(),
                        row_index=_row_idx(doc.metadata),
                    )
                )

        # deduplicate by organisation name
        seen: set[str] = set()
        unique_matches: List[SponsorMatch] = []
        for m in all_matches:
            if m.organisation_name not in seen:
                seen.add(m.organisation_name)
                unique_matches.append(m)

        return SponsorResponse(
            query=payload.question,
            sponsored=sponsored,
            status="Licensed" if sponsored else "Not Found",
            summary=response.content,
            matches=unique_matches,
            total_matches_found=len(unique_matches),
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
