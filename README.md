# LangChain RAG Pipeline with LlamaIndex (LlamaParse), Custom Embedding Container, Chroma DB, and Groq LLM

An enterprise-grade, microservice-based **Retrieval-Augmented Generation (RAG)** pipeline powered by:
- **LlamaIndex LlamaParse**: State-of-the-art document parsing (tables, formatting, layouts).
- **Custom Embedding Microservice Container (Docker)**: Dedicated microservice on port 8001 hosting `sentence-transformers/all-MiniLM-L6-v2`.
- **Local Persistent Chroma DB**: Fast vector indexing and similarity retrieval stored on local disk (`./chroma_data`).
- **LangChain + Groq (ChatGroq)**: Ultra-fast LLM answer generation using Groq's high-speed inference engine (`llama-3.3-70b-versatile` or `llama-3.1-8b-instant`).

---

## 🏗️ System Architecture

```
[ Ingest Document (PDF/DOCX/PPTX/TXT) ]     [ User Question ]
                    │                              │
                    ▼                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    LangChain RAG Pipeline                        │
│                                                                  │
│  1. LlamaParseLoader -> LlamaParse Cloud API (or local fallback) │
│     Converts complex tables & layout to Markdown Documents       │
│                                                                  │
│  2. CustomContainerEmbeddings -> Embed via Microservice          │
│     POST http://localhost:8001/embed (all-MiniLM-L6-v2)          │
│                                                                  │
│  3. Local Chroma Store -> Persistent Disk Indexing               │
│     Storage directory: ./chroma_data                             │
│                                                                  │
│  4. LCEL RAG Chain -> Context Formatting                         │
│                                                                  │
│  5. Groq ChatGroq -> Ultra-Fast Generation (Llama 3.3 70B)       │
└───────────────────────────┬──────────────────────────────────────┘
                            │ HTTP
                            ▼
     ┌──────────────────────────────────────────────┐
     │  Custom Embedding Service Container (Docker) │
     │  Port 8001 | sentence-transformers/MiniLM   │
     └──────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Start the Custom Embedding Docker Container
Spin up the containerized embedding microservice:

```bash
docker compose up -d --build
```

Verify the container is healthy:
```bash
docker compose ps
```
- **Embedding Microservice**: `http://localhost:8001/health`

---

### 2. Set Up the Python Client

Create or activate your Python virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

### 3. Launch the Streamlit Web Frontend 🌟

Run the interactive web interface for uploading PDFs, monitoring backend status, and chatting with Groq:

```bash
streamlit run streamlit_app.py
```
This opens `http://localhost:8501` in your browser where you can:
- Drag-and-drop PDFs, DOCX, TXT documents for **asynchronous parsing with LlamaParse**.
- View live health status of the Embedding container and Chroma indexed chunk counts.
- Chat with your documents using **Groq ChatGroq** (`llama-3.3-70b-versatile`).
- Expand chunk citations to view page numbers and source provenance.

---

### 4. Configure API Keys (`.env`)

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Configure your **Groq API Key** and **LlamaParse API Key**:
```ini
# 1. Groq LLM (Get key at https://console.groq.com)
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# 2. LlamaParse (Get key at https://cloud.llamaindex.ai)
LLAMA_CLOUD_API_KEY=llx_your_llamaparse_key_here
LLAMA_PARSE_RESULT_TYPE=markdown
```
*(Note: If no API keys are provided, the pipeline gracefully falls back to local document reading and a local extractive synthesizer for offline testing).*

---

### 4. Check Pipeline Readiness

Verify connectivity of all components:

```bash
python main.py check
```

---

### 5. Ingest a Document

Parse and index any document (PDF, DOCX, PPTX, TXT, HTML, markdown):

```bash
python main.py ingest sample_docs/sample.txt
```

This will:
1. Parse the document with **LlamaIndex LlamaParse** preserving structure and tables.
2. Convert and chunk the content into **LangChain Document** objects.
3. Generate 384-dimensional dense vectors using the **Custom Embedding Container** (`all-MiniLM-L6-v2`).
4. Persist vectors and metadata in the **local Chroma data store** (`./chroma_data`).

---

### 6. Ask Questions (RAG Query)

Ask questions and get answers powered by **Groq ChatGroq**:

```bash
python main.py query "What is the RPO and RTO for disaster recovery?"
```

Or run direct vector similarity search without LLM synthesis:

```bash
python main.py search "polyglot persistence database"
```

Or start an **interactive chat session**:

```bash
python main.py interactive
```

---

## ⚙️ Configuration Parameters (`.env`)

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | LLM engine (`groq`, `openai`, `ollama`, `mock`) |
| `GROQ_API_KEY` | `""` | API key from Groq Console |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model name on Groq (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, etc.) |
| `LLAMA_CLOUD_API_KEY` | `""` | LlamaParse API Key from LlamaIndex Cloud |
| `LLAMA_PARSE_RESULT_TYPE` | `markdown` | Parsing output format (`markdown` or `text`) |
| `EMBEDDING_API_URL` | `http://localhost:8001` | Custom embedding container URL |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma_data` | Directory where Chroma persists vector files |
| `CHROMA_COLLECTION_NAME` | `rag_documents` | Collection name in Chroma |
| `CHUNK_SIZE` | `500` | Target chunk size in characters |
| `CHUNK_OVERLAP` | `50` | Chunk overlap in characters |
| `RETRIEVAL_TOP_K` | `4` | Number of chunks retrieved for answers |

---

---

## 🐝 Docker Swarm Automated Scaling (1 ➔ 3 ➔ 1 Replicas)

The embedding service microservice features **automated elastic scaling** on Docker Swarm:
1. **Idle State**: Runs **1 replica** by default to conserve system resources.
2. **Auto-Scale Trigger**: When ingesting a document that produces **>= 50 or 100 chunks**, the pipeline automatically scales the service to **3 replicas** (`docker service scale rag_stack_embedding-service=3`).
3. **Load Balancing**: The Swarm Ingress routing mesh automatically distributes incoming batches across all 3 replicas.
4. **Auto Scale-Down**: Once ingestion and indexing finish, the pipeline automatically scales the service back down to **1 replica**.

### Deploy the Stack:
```powershell
.\deploy_swarm.ps1
```
Or manually:
```bash
docker build -t custom-embedding-service:latest ./embedding_service
docker stack deploy -c docker-compose.yml rag_stack
```

Verify service status:
```bash
docker service ls
```

---

## 🧪 100+ Chunk Verification Benchmark

Run the full end-to-end flow with a 100+ chunk document:

```bash
python main.py test-100
```
Or run the automated unit/integration test suite:
```bash
python test_e2e_flow.py
```
This tests:
1. Parsing a multi-section document into **100+ chunks**.
2. Batch embedding chunks across the **3 Swarm replicas**.
3. Storing and indexing in **local Chroma DB**.
4. Retrieving relevant context with similarity scores and querying **Groq ChatGroq**.

---

## 🧩 Programmatic Python Usage

```python
from rag import CustomContainerEmbeddings, LlamaParseLoader, ChromaStoreManager, RAGPipeline

# Initialize RAG pipeline
pipeline = RAGPipeline()

# Ingest document with LlamaParse
pipeline.ingest_file("sample_docs/large_knowledge_base.txt")

# Query pipeline with Groq ChatGroq
result = pipeline.query("What are the disaster recovery requirements?")
print("Answer:", result["answer"])
print("Sources:", result["sources"])
```

---

## 🐳 Stopping the Services

To remove the Swarm stack:
```bash
docker stack rm rag_stack
```
Or if using standard Docker Compose:
```bash
docker compose down
```
