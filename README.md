# UK Visa Sponsor Checker

A full-stack RAG application upgraded with a ReAct AI Agent that lets job seekers instantly check whether a UK company holds a valid Skilled Worker sponsorship licence - without manually searching the 141,000-row Home Office register.

## The Problem

International job seekers in the UK waste hours cross-checking companies from job boards against the government sponsor register. Company names on job posts rarely match registered names exactly - abbreviations, symbols, hyphens, and initials cause manual searches to fail silently. Checking 15 companies manually takes 3 hours. This app does it in seconds.

## The Solution

### Three-layer search pipeline
1. **Query normalisation** - handles symbols ('%', '&', '+'), abbreviations ('Ltd -> Limited'), and number variants ('1-2 -> 1 to 2') before searching
2. **Semantic search** - sentence-transformers embeddings + ChromaDB retrieve the most contextually similar sponsor records using MMR (Maximum Marginal Relevance)
3. **Fuzzy fallback** - RapidFuzz token matching catches character-level variations the embedding model misses (e.g. 'I Net Software' -> 'I-Net Software Solutions')

### ReAct AI Agent layer
On top of the RAG pipeline, a ReAct (Reason + Act) agent adds multi-tool reasoning:

'''
User: "I have an interview at Deloitte - can they sponsor me?"
    ↓
Thought: I need to check the sponsor register first
Action: SponsorRegisterCheck -> searches ChromaDB (RAG pipeline)
Result: Deloitte LLP | London | Skilled Worker | A rating
    ↓
Thought: I should check recent hiring news too
Action: WebSearch -> DuckDuckGo live web search
Result: Recent Deloitte hiring announcements...
    ↓
Final Answer: "Deloitte LLP holds an A-rated Skilled Worker licence
in London. Based on recent news, they actively sponsor international
candidates for technology and consulting roles."
'''

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) - free, local |
| Vector DB | ChromaDB (persisted to disk) |
| Fuzzy matching | RapidFuzz |
| LLM | Groq API - llama-3.1-8b-instant (free tier) |
| Agent tools | SponsorRegisterCheck + DuckDuckGo WebSearch |
| Frontend | Angular 17+ |
| Data | UK Home Office Register of Licensed Sponsors (141,026 rows) |

## Features

- '/ask' - RAG endpoint: normalise -> semantic search -> fuzzy fallback -> Groq answer
- '/agent/query' - ReAct Agent endpoint: sponsor register + web search + reasoning trace
- '/ingest-xlsx' - upload any CSV or XLSX to re-index
- Structured JSON response - sponsored status, rating, town, visa route, confidence
- Reasoning steps returned - see exactly how the agent thinks
- Angular frontend with two tabs - Sponsor Search and AI Agent
- Zero cost to run - free embeddings, free Groq tier, free ChromaDB

## Quick Start

'''bash
# 1. Clone and install
git clone https://github.com/manikantakallakuri8-bit/uk-visa-sponsor-checker.git
cd uk-visa-sponsor-checker
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 3. Pre-load the sponsor register into ChromaDB (run once - takes 5-15 mins)
python ingest.py

# 4. Verify ingestion worked
python verify_chroma.py

# 5. Start the API
python -m uvicorn main:app --reload
'''

Frontend:
'''bash
cd frontend
npm install
ng serve
# Open http://localhost:4200
'''

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| '/health' | GET | API status and ChromaDB doc count |
| '/ask' | POST | RAG-based sponsor check |
| '/agent/query' | POST | ReAct agent - register + web search |
| '/ingest-xlsx' | POST | Upload CSV or XLSX to re-index |

### /ask Example

'''bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Does Tata Consultancy Services sponsor Skilled Workers?\", \"top_k\": 8}"
'''

Response:
'''json
{
  "query": "Does Tata Consultancy Services sponsor Skilled Workers?",
  "sponsored": true,
  "status": "Licensed",
  "summary": "Tata Consultancy Services Ltd holds an A-rated Skilled Worker licence in London.",
  "matches": [
    {
      "organisation_name": "Tata Consultancy Services Ltd",
      "town": "London",
      "route": "Skilled Worker",
      "rating": "Worker (A rating)",
      "row_index": 112847
    }
  ],
  "total_matches_found": 1,
  "data_source": "UK Home Office Register of Licensed Sponsors",
  "disclaimer": "Always verify at gov.uk before applying"
}
'''

### /agent/query Example

'''bash
curl -X POST "http://127.0.0.1:8000/agent/query" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"I have an interview at Deloitte next week - can they sponsor me?\"}"
'''

Response:
'''json
{
  "question": "I have an interview at Deloitte next week - can they sponsor me?",
  "final_answer": "Deloitte LLP holds an A-rated Skilled Worker licence in London.",
  "reasoning_steps": [
    "Thought: I need to check the UK sponsor register first.",
    "Tool: SponsorRegisterCheck | Input: Deloitte",
    "Result: Deloitte LLP | London | Skilled Worker | Worker (A rating)",
    "Thought: I should also search the web for recent hiring news.",
    "Tool: WebSearch | Input: Deloitte UK Skilled Worker visa sponsorship",
    "Result: Deloitte announces...",
    "Thought: I now have register data and web context. Generating final answer."
  ],
  "tools_used": ["SponsorRegisterCheck", "WebSearch"],
  "data_source": "UK Home Office Register + Web Search",
  "disclaimer": "Always verify sponsorship at gov.uk before applying"
}
'''

## Architecture

'''
tier2.csv (141k rows)
    ↓ python ingest.py (run once)
sentence-transformers -> vectors -> ChromaDB (persisted)

User query
    ↓
normalise_company_name()     strip symbols, abbreviations, numbers
    ↓              ↓
ChromaDB MMR      fuzzy_fallback()
semantic search   RapidFuzz on raw CSV
    ↓              ↓
    └── merge results ──┘
            ↓
    /ask -> Groq LLM -> structured JSON
    /agent/query -> ReAct loop -> register + web -> Groq -> answer + trace
'''

## Limitations

- **Data freshness** - register updates monthly, re-run 'python ingest.py --reset' to refresh
- **Trading names** - "Boots" won't find "The Boots Company PLC" without Companies House API
- **Always verify** results at [gov.uk](https://www.gov.uk/government/publications/register-of-licensed-sponsors-workers) before making job application decisions

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| 'GROQ_API_KEY' | required | Your Groq API key from console.groq.com |
| 'GROQ_MODEL' | 'llama-3.1-8b-instant' | Groq model name |
| 'EMBEDDING_MODEL' | 'sentence-transformers/all-MiniLM-L6-v2' | Local embedding model |
| 'CHROMA_PERSIST_DIRECTORY' | './chroma_db' | Where ChromaDB stores vectors |
| 'TIER2_CSV_PATH' | 'tier2.csv' | Path to sponsor register CSV |
| 'RESET_COLLECTION_ON_INGEST' | 'true' | Wipe and rebuild on each ingest |
