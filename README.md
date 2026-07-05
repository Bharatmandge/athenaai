# 🦉 Athena — Multi-Agent GraphRAG Knowledge Intelligence Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-purple?style=for-the-badge)
![Neo4j](https://img.shields.io/badge/Neo4j-Knowledge--Graph-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector--Store-DC143C?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**A production-grade Multi-Agent GraphRAG platform that combines semantic vector search with knowledge graph traversal to deliver cited, self-correcting answers from your documents.**

[Live Demo](#) · [API Docs](#) · [Report Bug](#) · [Request Feature](#)

</div>

---

## 📌 What is Athena?

Athena is not a basic RAG chatbot. It is a **Multi-Agent Knowledge Intelligence Platform** that:

- Ingests documents (PDF, DOCX, TXT) and builds both a **vector store** (Qdrant) and a **knowledge graph** (Neo4j) simultaneously
- Answers queries using **GraphRAG** — merging semantic similarity search with graph relationship traversal in parallel
- Routes every query through a **5-node LangGraph agent pipeline**: Planner → Retriever → Graph Agent → Responder → Critic
- **Self-corrects** — a Critic Agent scores every answer (0.0–1.0) and triggers automatic retry if quality is below threshold
- Returns **cited answers** with source documents, query entities, agent execution logs, and critique scores
- Falls back from **Gemini → Groq** automatically on quota limits — zero downtime

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Frontend                           │
│          Chat · Graph Explorer · Research · Evaluation          │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP / SSE
┌─────────────────────▼───────────────────────────────────────────┐
│                     FastAPI Backend                             │
│         /upload · /chat · /search · /graph · /research         │
└──────┬──────────────────────────────────────┬───────────────────┘
       │                                      │
┌──────▼──────────────────┐    ┌──────────────▼──────────────────┐
│   LangGraph StateGraph  │    │      Ingestion Pipeline         │
│                         │    │                                 │
│  [Planner Agent]        │    │  Upload → Parse → Chunk         │
│       ↓                 │    │       ↓          ↓              │
│  [Retriever Agent]      │    │  Embed (Gemini)  Extract        │
│       ↓                 │    │       ↓          ↓              │
│  [Graph Agent]          │    │  Qdrant Cloud  Neo4j Aura       │
│       ↓                 │    │  (vectors)     (entities)       │
│  [Responder Agent]      │    └─────────────────────────────────┘
│       ↓                 │
│  [Critic Agent]─────────┤ score < 0.7 → retry
│       ↓                 │
│     END                 │
└─────────────────────────┘
       │                  │
┌──────▼──────┐   ┌───────▼──────┐
│ Qdrant Cloud│   │  Neo4j Aura  │
│ Vector Store│   │Knowledge Graph│
└─────────────┘   └──────────────┘
       │                  │
┌──────▼──────────────────▼──────┐
│     Gemini API (primary)       │
│     Groq API   (fallback)      │
└────────────────────────────────┘
```

---

## ✨ Key Features

### 🔍 GraphRAG Hybrid Retrieval
Combines Qdrant vector similarity search with Neo4j graph traversal running **in parallel** via `ThreadPoolExecutor`. Answers are enriched with relationship triples (e.g. `DentalVision AI --[USES]--> PyTorch`) that plain RAG systems cannot surface.

### 🤖 5-Node LangGraph Multi-Agent Pipeline
Every query passes through a compiled `StateGraph` with typed state:

| Agent | Role |
|---|---|
| **Planner** | Generates a 3-step reasoning plan for the query |
| **Retriever** | Runs hybrid Qdrant + Neo4j retrieval in parallel |
| **Graph Agent** | Deep 3-hop Neo4j traversal + shortest path between entities |
| **Responder** | Generates answer using plan + merged context |
| **Critic** | Scores answer 0.0–1.0, triggers retry if score < 0.7 |

### 🔄 Self-Correcting Critic Loop
The Critic Agent evaluates every draft answer on **Faithfulness** (0–0.5) and **Completeness** (0–0.5). If the combined score falls below 0.7, the critique is injected back into the Responder for one automatic retry.

### ⚡ Gemini → Groq Automatic Fallback
Every LLM call is wrapped with automatic fallback:
- **Primary**: Gemini 2.0 Flash
- **Fallback**: Groq `llama-3.3-70b-versatile`

Zero manual intervention — if Gemini hits quota, Groq handles it silently.

### 📊 Full Observability
Every API response includes:
- `agent_logs` — each agent's name, duration, and status
- `critique_score` — Critic's quality score (0.0–1.0)
- `critique` — specific feedback on answer quality
- `plan` — Planner's reasoning steps
- `graph_context_used` — whether Neo4j contributed
- `model_used` — which LLM actually generated the answer

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Agent Orchestration** | LangGraph (StateGraph, conditional edges) |
| **Backend** | FastAPI, Python 3.10+, Uvicorn |
| **Frontend** | React 18, Vite, Tailwind CSS, Shadcn UI |
| **Vector Store** | Qdrant Cloud (text-embedding-004, 768-dim, cosine) |
| **Knowledge Graph** | Neo4j Aura (Cypher, MERGE, variable-length paths) |
| **Primary LLM** | Google Gemini 2.0 Flash |
| **Fallback LLM** | Groq llama-3.3-70b-versatile |
| **Embeddings** | Google text-embedding-004 (retrieval_document / retrieval_query) |
| **Document Parsing** | PyPDF2, python-docx, LangChain RecursiveCharacterTextSplitter |
| **Graph Visualization** | React Flow |
| **Evaluation** | RAGAS (Faithfulness, Answer Relevancy, Context Precision) |
| **Tracing** | LangSmith |
| **Containerization** | Docker, docker-compose |
| **Deployment** | Vercel (frontend), Railway (backend) |
| **Database** | SQLite (document metadata) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Git

### API Keys Required

| Service | Purpose | Free Tier |
|---|---|---|
| [Google AI Studio](https://aistudio.google.com) | Gemini LLM + Embeddings | ✅ Yes |
| [Groq Console](https://console.groq.com) | Fallback LLM | ✅ Yes |
| [Qdrant Cloud](https://cloud.qdrant.io) | Vector store | ✅ Yes (1GB) |
| [Neo4j Aura](https://neo4j.com/cloud/aura) | Knowledge graph | ✅ Yes (200MB) |
| [Tavily](https://tavily.com) | Web search (Research Mode) | ✅ Yes |
| [LangSmith](https://smith.langchain.com) | Agent tracing | ✅ Yes |

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/yourusername/athena-graphrag-platform.git
cd athena-graphrag-platform
```

**2. Backend setup**

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**3. Environment variables**

Create `backend/.env`:

```env
# LLM APIs
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key

# Vector Store
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# Knowledge Graph
NEO4J_URI=bolt+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Web Search (Research Mode)
TAVILY_API_KEY=your_tavily_api_key

# Tracing
LANGSMITH_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=athena-graphrag
```

**4. Start backend**

```bash
uvicorn backend.main:app --reload
# API running at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

**5. Frontend setup**

```bash
cd frontend
npm install
npm run dev
# Frontend running at http://localhost:5173
```

---

## 🐳 Docker

```bash
# Build and run with docker-compose
docker-compose up --build

# Backend: http://localhost:8000
# Frontend: http://localhost:5173
```

---

## 📡 API Reference

### Upload Document
```http
POST /api/upload
Content-Type: multipart/form-data

file: <PDF | DOCX | TXT>
```

Response:
```json
{
  "doc_id": "uuid",
  "chunks": 42,
  "status": "graph_built"
}
```

### Chat (GraphRAG)
```http
POST /api/chat/
Content-Type: application/json

{
  "query": "What deep learning architecture does DentalVision AI use?",
  "top_k": 5
}
```

Response:
```json
{
  "answer": "DentalVision AI uses the 3D Res-UNet architecture...",
  "citations": ["DentalVision AI.pdf"],
  "query_entities": ["DentalVision AI"],
  "graph_context_used": true,
  "model_used": "groq/llama-3.3-70b",
  "critique_score": 0.9,
  "critique": "Answer is accurate and complete",
  "plan": "1. Research architecture...",
  "agent_logs": [
    {"agent": "planner",     "duration_s": 1.1,  "status": "done"},
    {"agent": "retriever",   "duration_s": 2.0,  "status": "done", "chunks_found": 5},
    {"agent": "graph_agent", "duration_s": 0.2,  "status": "done", "triples_found": 19},
    {"agent": "responder",   "duration_s": 2.1,  "status": "done"},
    {"agent": "critic",      "duration_s": 1.1,  "status": "done", "score": 0.9}
  ]
}
```

### Semantic Search
```http
POST /api/search
Content-Type: application/json

{
  "query": "neural network architecture",
  "top_k": 5
}
```

### Graph Explorer
```http
GET /api/graph/{entity_name}?depth=2
```

Response: React Flow compatible `{ nodes: [...], edges: [...] }`

### Collection Stats
```http
GET /api/search/count
```

---

## 🧠 How GraphRAG Works

Plain RAG retrieves text chunks similar to your query. GraphRAG does more:

```
Query: "What technologies does DentalVision AI use?"

Plain RAG:
→ Find chunks containing "DentalVision AI technologies"
→ Return text snippets

GraphRAG (Athena):
→ Extract entities: ["DentalVision AI"]
→ Vector search: top-5 semantic chunks         ← Qdrant
→ Graph traversal: DentalVision AI neighbors   ← Neo4j
   DentalVision AI --[MENTIONS]--> PyTorch
   DentalVision AI --[MENTIONS]--> FastAPI
   DentalVision AI --[MENTIONS]--> Docker
   DentalVision AI --[USES]--> 3D Res-UNet
   ...19 triples total
→ Merge both contexts → send to LLM
→ Answer lists 16 specific technologies with relationships explained
```

---

## 📁 Project Structure

```
athena-graphrag-platform/
├── backend/
│   ├── agents/
│   │   ├── state.py              # AthenaState TypedDict
│   │   ├── planner_agent.py      # Query planning
│   │   ├── retriever_agent.py    # Hybrid retrieval
│   │   ├── graph_agent.py        # Deep Neo4j traversal
│   │   ├── responder_agent.py    # Answer generation
│   │   ├── critic_agent.py       # Quality scoring + retry
│   │   └── fallback_agent.py     # No-context handler
│   ├── graph/
│   │   ├── workflow.py           # LangGraph StateGraph
│   │   ├── neo4j_client.py       # Driver + query helper
│   │   ├── entity_extractor.py   # Gemini entity extraction
│   │   ├── graph_builder.py      # Cypher MERGE operations
│   │   └── graph_retriever.py    # Read-only Cypher queries
│   ├── retrieval/
│   │   ├── hybrid_retriever.py   # Parallel Qdrant + Neo4j
│   │   └── context_builder.py    # Merge + format context
│   ├── services/
│   │   ├── embedder.py           # Gemini text-embedding-004
│   │   ├── vector_store.py       # Qdrant operations
│   │   ├── llm_service.py        # Gemini → Groq fallback
│   │   ├── document_parser.py    # PDF/DOCX/TXT parsing
│   │   └── chunker.py            # RecursiveCharacterTextSplitter
│   ├── routes/
│   │   ├── upload.py             # POST /upload
│   │   ├── chat.py               # POST /chat
│   │   ├── search.py             # POST /search
│   │   └── graph.py              # GET /graph/{node}
│   ├── models/
│   │   └── document.py           # SQLAlchemy Document model
│   └── main.py                   # FastAPI app + startup
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Chat.jsx          # GraphRAG chat interface
│   │   │   ├── GraphExplorer.jsx # React Flow visualization
│   │   │   ├── Research.jsx      # Research mode
│   │   │   └── Evaluation.jsx    # RAGAS dashboard
│   │   └── App.jsx
│   └── package.json
├── architecture/
│   └── system_design.md
├── docs/
│   ├── api_contract.md
│   └── roadmap.md
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## 🧪 Evaluation

Athena uses **RAGAS** for objective RAG evaluation:

| Metric | Description |
|---|---|
| **Faithfulness** | Are all claims supported by retrieved context? |
| **Answer Relevancy** | Does the answer address the actual question? |
| **Context Precision** | Is the retrieved context relevant to the query? |
| **Context Recall** | Did retrieval capture all necessary information? |

Every answer also includes an inline **Critic score** (0.0–1.0) computed in real-time by the Critic Agent.

---

## 🗺️ Roadmap

- [x] Document ingestion pipeline (PDF, DOCX, TXT)
- [x] Qdrant vector store with Gemini embeddings
- [x] Neo4j knowledge graph with entity extraction
- [x] GraphRAG hybrid retrieval (parallel)
- [x] LangGraph 5-node multi-agent pipeline
- [x] Critic Agent with self-correcting retry loop
- [x] Gemini → Groq automatic fallback
- [x] Agent execution logs in every response
- [ ] Research Mode (Tavily web search + report generation)
- [ ] RAGAS evaluation dashboard
- [ ] React Flow graph explorer UI
- [ ] Streaming SSE responses in frontend
- [ ] Docker + production deployment

---

## 👤 Author

**Bharat** — MCA Final Year · AI/ML Engineer

[![LinkedIn](https://www.linkedin.com/in/bharat-mandge/)
[![GitHub](https://github.com/Bharatmandge)


---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <strong>Built with LangGraph · Neo4j · Qdrant · Gemini · Groq · FastAPI · React</strong>
  <br/>
  <em>Production-grade Multi-Agent GraphRAG — not just another chatbot</em>
</div>
