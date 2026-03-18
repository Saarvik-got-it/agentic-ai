# RAG Agent - Production-Ready Document-Based Question Answering

A scalable, modular Retrieval-Augmented Generation (RAG) system with a pipeline-ready Content Agent, built with LangChain components, Google Gemini, and FAISS. Designed for standalone CLI usage and seamless integration into backend workflows.

## ✨ Features

### Core Capabilities
- **📄 Document Loading**: Automatic loading of PDF and TXT files from a configurable directory
- **🧬 Smart Chunking**: RecursiveCharacterTextSplitter with configurable chunk size and overlap
- **🔍 Vector Embeddings**: Direct Google Generative AI SDK embeddings (Gemini) with automatic vector storage
- **📦 FAISS Vector Database**: Persistent local vector store for fast semantic search
- **🤖 LLM Response Generation**: Google Gemini with intelligent fallback mechanism
- **✍️ Content Transformation**: Config-driven Content Agent for persona-based, structured output generation
- **♻️ Caching**: Automatic reuse of existing vector indices (no recomputation)
- **⏱️ Performance Tracking**: Built-in timing and latency measurement
- **📊 Comprehensive Logging**: Detailed logs for debugging and monitoring

### Production Features
- ✅ Environment-based configuration (`.env` files)
- ✅ Modular architecture for easy component swapping
- ✅ Error handling and graceful degradation
- ✅ Vector store abstraction (FAISS → MongoDB/Pinecone ready)
- ✅ Batch processing capabilities
- ✅ CLI interface for standalone usage
- ✅ FastAPI-ready structure for backend integration
- ✅ No hardcoded values or paths

## 🏗️ Project Structure

```
agentic-ai/
├── agents/                      # RAG agent implementations
│   ├── __init__.py
│   └── rag_agent.py            # Main RAG Agent class
│
├── config/
│   └── content_agent_config.json  # Personas/content types/prompt templates
│
├── utils/                       # Shared utilities and abstractions
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── logger.py               # Logging setup
│   ├── document_loader.py      # Document loading (PDF/TXT)
│   ├── embeddings.py           # Direct Gemini SDK embeddings wrapper
│   ├── vector_store.py         # FAISS abstraction layer
│   └── llm.py                  # LLM with fallback mechanism
│
├── pipelines/                   # Data processing pipelines
│   ├── __init__.py
│   └── ingestion.py            # Document ingestion pipeline
│
├── data/
│   └── documents/              # Place your PDF/TXT files here
│
├── vector_store/               # FAISS index storage (auto-created)
│
├── logs/                        # Application logs (auto-created)
│
├── main.py                     # CLI entry point
├── content_agent.py            # Standalone Content Agent (CLI + function)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── README.md                   # This file
```

## 🔗 Multi-Agent Pipeline

Typical flow for multi-agent orchestration:

`User Query -> RAG Agent -> Content Agent -> Email Agent`

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10 or higher
- Google Gemini API key (obtain from [Google AI Studio](https://aistudio.google.com/apikey))

### 2. Installation

```bash
# Clone or navigate to project
cd agentic-ai

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Create .env file from template
cp .env.example .env

# Edit .env and add your Google API key
# GOOGLE_API_KEY=your_actual_key_here
```

### 4. Add Documents

Place your PDF and TXT files in the `data/documents/` folder:

```bash
mkdir -p data/documents
cp your_files.pdf data/documents/
cp your_files.txt data/documents/
```

### 5. Run RAG Agent

#### Interactive Mode
```bash
python main.py
```

Then ask questions:
```
📝 Query: What are the main topics covered?
```

#### Single Query
```bash
python main.py -q "Your question here"
```

#### Document Ingestion (explicit)
```bash
python main.py --ingest
```

## ✍️ Content Agent

The Content Agent transforms factual input (from RAG or any source) into structured, persona-driven content.

Core function:

```python
from content_agent import content_agent

result = content_agent(rag_output, "blog", "technical_writer")
```

Design highlights:
- Config-driven personas, content types, and prompt templates (no hardcoded prompt logic)
- Dynamic prompt construction with clarity and no-hallucination rules
- Standalone CLI support and pipeline-ready function interface
- Uses `utils/llm.py -> generate_with_fallback()` for generation
- Logging of input size, persona, content type, and output size
- Error handling for empty input, invalid persona/content type, and runtime failures

CLI examples:

```bash
python content_agent.py --list-options

python content_agent.py \
    --input-text "Cristiano Ronaldo is a Portuguese footballer..." \
    --content-type summary \
    --persona research_analyst

python content_agent.py \
    --input-file ./data/facts.txt \
    --content-type report \
    --persona technical_writer
```

Optional config override:

```bash
set CONTENT_AGENT_CONFIG=./config/content_agent_config.json
```

## 📋 Configuration

Configure via `.env` file:

```env
# API
GOOGLE_API_KEY=your_key

# Paths
DOCUMENT_FOLDER=./data/documents
VECTOR_STORE_PATH=./vector_store/faiss_index

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/rag_agent.log

# Retrieval
TOP_K_CHUNKS=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# LLM
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=512
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GOOGLE_API_KEY` | Required | Your Google Gemini API key |
| `DOCUMENT_FOLDER` | `./data/documents` | Where to load documents from |
| `VECTOR_STORE_PATH` | `./vector_store/faiss_index` | Where to store FAISS index |
| `TOP_K_CHUNKS` | 5 | Number of relevant documents to retrieve |
| `CHUNK_SIZE` | 1000 | Characters per text chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `LLM_TEMPERATURE` | 0.3 | LLM creativity (0.0=deterministic, 1.0=random) |
| `LLM_MAX_TOKENS` | 512 | Maximum response length |

## 🔧 Architecture

### Core Components

#### 1. **RAG Agent** (`agents/rag_agent.py`)
Main orchestrator class that coordinates the entire pipeline:

```python
from agents.rag_agent import RAGAgent

# Initialize
agent = RAGAgent()

# Query
response = agent.query("Your question")
print(response)
```

Or use the convenience function:

```python
from agents.rag_agent import rag_agent

response = rag_agent("Your question")
```

#### 2. **Document Loader** (`utils/document_loader.py`)
- Automatically finds and loads PDF and TXT files
- Extracts text and metadata
- Splits documents into manageable chunks

```python
from utils.document_loader import load_documents, split_documents

docs = load_documents("./data/documents")
split_docs = split_documents(docs, chunk_size=1000)
```

#### 3. **Embeddings** (`utils/embeddings.py`)
- Uses `google.generativeai` directly for embedding generation
- Converts text to 768-dimensional vectors
- Includes model-name fallback (`embedding-001` -> `models/embedding-001`)
- Exposes a LangChain-compatible interface for FAISS integration

```python
from utils.embeddings import embed_text, embed_texts

vector = embed_text("Sample text")
vectors = embed_texts(["text1", "text2"])
```

Model used for embeddings:
- Primary: `embedding-001`
- Fallback: `models/embedding-001`
- Runtime fallback: API-discovered embed-capable models (for example `models/gemini-embedding-001`)

#### 4. **Vector Store** (`utils/vector_store.py`)
- FAISS-based abstraction layer
- Handles index creation and persistence
- Provides similarity search interface
- **Abstraction allows easy migration** to MongoDB Atlas, Pinecone, etc.

```python
from utils.vector_store import VectorStoreManager

manager = VectorStoreManager()
manager.get_or_create(documents)
docs = manager.retrieve_documents("query", k=5)
```

#### 5. **LLM with Fallback** (`utils/llm.py`)
Implements automatic retry across multiple Gemini models:

**Model Priority (High → Low):**
1. Gemini 3.1 Flash Lite
2. Gemini 2.5 Flash Lite
3. Gemini 3 Flash
4. Gemini 2.5 Flash

**Fallback Triggers:**
- Rate limit (429)
- Quota exceeded
- Timeout
- Service unavailable (503)

```python
from utils.llm import generate_with_fallback

response = generate_with_fallback(
    prompt="Your prompt",
    system_prompt="Optional context"
)
```

#### 6. **Configuration** (`utils/config.py`)
Centralized settings management with validation:

```python
from utils.config import get_settings

settings = get_settings()
print(settings.google_api_key)
print(settings.top_k_chunks)
```

#### 7. **Logging** (`utils/logger.py`)
Structured logging to both console and file:

```python
from utils.logger import setup_logger

logger = setup_logger(__name__)
logger.info("Processing started")
```

#### 8. **Document Ingestion Pipeline** (`pipelines/ingestion.py`)
Batch processing of documents with monitoring:

```python
from pipelines.ingestion import ingest_documents, reindex_documents

# Load/create index
ingest_documents()

# Force recreate
reindex_documents()
```

## 🔄 Workflow

```
┌─────────────┐
│   USER      │
│  QUERY      │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│    RAG AGENT                    │
│  ( agents/rag_agent.py )        │
└──────┬──────────────────────────┘
       │
       ├─► [1] Load Documents (if first run)
       │       └─► utils/document_loader.py
       │           ├─ Find PDF/TXT files
       │           ├─ Extract text
       │           └─ Split into chunks
       │
       ├─► [2] Create Embeddings
       │       └─► utils/embeddings.py
       │           └─ Google Gemini API
       │
       ├─► [3] Index with FAISS
       │       └─► utils/vector_store.py
       │           └─ Save locally (reuse)
       │
       ├─► [4] Retrieve Context
       │       └─► Query: similarity_search(k=5)
       │           └─ Get top-5 relevant chunks
       │
       ├─► [5] Generate Response
       │       └─► utils/llm.py
       │           ├─ Try Model 1
       │           ├─ Fallback → Model 2
       │           ├─ Fallback → Model 3
       │           └─ Fallback → Model 4
       │
       └─► [6] Return Answer
           └─ Formatted Response
```

## 🧠 LLM Fallback Mechanism

The system automatically handles failures:

```python
# Automatic fallback flow:
Try Model 1 (Gemini 3.1 Flash Lite)
  ├─ Success? → Return response
  └─ Rate Limit/Timeout? → Try Model 2

Try Model 2 (Gemini 2.5 Flash Lite)
  ├─ Success? → Return response
  └─ Rate Limit/Timeout? → Try Model 3

Try Model 3 (Gemini 3 Flash)
  ├─ Success? → Return response
  └─ Rate Limit/Timeout? → Try Model 4

Try Model 4 (Gemini 2.5 Flash)
  ├─ Success? → Return response
  └─ All Failed? → Return graceful error

# Logged example:
[INFO] Attempting to generate response with model: gemini-3.1-flash-lite
[WARNING] Model gemini-3.1-flash-lite failed (rate limit): Rate limit exceeded
[INFO] Attempting to generate response with model: gemini-2.5-flash-lite
[INFO] Successfully generated response using gemini-2.5-flash-lite
```

## 🔌 Integration Examples

### FastAPI Backend

```python
# app.py
from fastapi import FastAPI
from agents.rag_agent import rag_agent

app = FastAPI()

@app.post("/ask")
async def ask_question(query: str):
    response = rag_agent(query)
    return {"query": query, "answer": response}

# Run: uvicorn app:app --reload
```

### Batch Processing

```python
from agents.rag_agent import RAGAgent

agent = RAGAgent()

queries = [
    "What is the main topic?",
    "How does it work?",
    "What are the benefits?"
]

for query in queries:
    answer = agent.query(query)
    print(f"Q: {query}\nA: {answer}\n")
```

### Custom Pipeline

```python
from utils.document_loader import load_documents, split_documents
from utils.vector_store import VectorStoreManager
from utils.llm import generate_with_fallback

# Custom workflow
docs = load_documents("./my_docs")
split = split_documents(docs)
manager = VectorStoreManager()
manager.create_vector_store(split)
retrieved = manager.retrieve_documents("my query", k=3)
context = "\n".join([d.page_content for d in retrieved])
answer = generate_with_fallback(f"Based on: {context}")
```

## 📊 Logging Output

The system logs to both console and file with detailed information:

```
2024-03-18 14:23:45 - agents.rag_agent - INFO - Processing query: What is machine learning?
2024-03-18 14:23:45 - utils.document_loader - INFO - Loading documents from ./data/documents
2024-03-18 14:23:45 - utils.document_loader - INFO - Found 3 PDF files and 2 TXT files
2024-03-18 14:23:46 - utils.document_loader - INFO - Loading PDF: document1.pdf
2024-03-18 14:23:47 - utils.embeddings - INFO - Initializing Google Gemini embeddings via direct SDK
2024-03-18 14:23:48 - utils.vector_store - INFO - Creating vector store from 15 documents
2024-03-18 14:23:52 - utils.vector_store - INFO - Vector store created and saved in 4.23s
2024-03-18 14:23:53 - utils.llm - INFO - Attempting to generate response with model: gemini-3.1-flash-lite
2024-03-18 14:23:55 - utils.llm - INFO - Successfully generated response using gemini-3.1-flash-lite
2024-03-18 14:23:55 - agents.rag_agent - INFO - Query processed in 10.25s

Logs saved to: ./logs/rag_agent_20240318.log
```

## 🛡️ Error Handling

The system provides graceful error handling:

```python
# Missing API key
ValueError: Configuration error: GOOGLE_API_KEY not set

# No documents
"No documents found in the knowledge base. Please add documents to the data/documents folder."

# All LLM models fail
"Unable to generate response. All Gemini models failed. Please try again later."

# Query not found in context
"Not enough information provided in the documents."
```

## 🔐 Security Considerations

- **API Keys**: Never commit `.env` files. Use `.env.example` as template.
- **Document Folder**: Ensure proper permissions on sensitive documents.
- **Vector Store**: FAISS index is unencrypted. Store in secure location for production.
- **Logging**: Remove sensitive data from logs if needed.

## 🎯 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

### Environment Setup

```bash
# .env (production)
GOOGLE_API_KEY=prod_key
DOCUMENT_FOLDER=/mnt/documents
VECTOR_STORE_PATH=/mnt/vector_store
LOG_LEVEL=WARNING
```

### Monitoring

```python
# Check agent status
python main.py --verbose
tail -f logs/rag_agent_*.log
```

## 🔄 Updating Documents

To add new documents and rebuild the index:

```bash
# 1. Copy documents to data/documents/
cp new_docs.pdf data/documents/

# 2. Reindex
python main.py --ingest

# Or force rebuild:
# python pipelines/ingestion.py --force
```

## 🧪 Testing & Validation

```python
# Simple validation
from agents.rag_agent import rag_agent

# Test basic query
response = rag_agent("What is the main topic?")
assert len(response) > 10, "Response too short"
assert "Not enough information" not in response or "Topic" in response

print("✅ Basic test passed")
```

## 📈 Performance Characteristics

Typical performance on modern hardware:

| Operation | Time (ms) | Notes |
|-----------|-----------|-------|
| Document Loading (5 files) | 500-1000 | One-time, cached |
| Embedding Generation | 100-200 | Per chunk |
| Vector Index Creation | 2000-5000 | One-time, reused |
| Query Retrieval | 50-100 | Fast semantic search |
| LLM Response Generation | 2000-4000 | Network-dependent |
| **Total Query** | **3000-5000** | End-to-end latency |

## 🚀 Advanced Customization

### Custom Vector Store (MongoDB Atlas)

```python
# utils/vector_store.py - Replace FAISS with MongoDB
class MongoVectorStore:
    def create_vector_store(self, documents):
        # MongoDB Atlas implementation
        pass

# agents/rag_agent.py - Use MongoDB instead
manager = MongoVectorStore()
```

### Custom LLM (OpenAI, Anthropic)

```python
# utils/llm.py - Replace Gemini
from langchain_openai import ChatOpenAI

def get_llm_with_fallback():
    return ChatOpenAI(model="gpt-4")
```

### Custom Prompt Template

```python
# utils/llm.py - Modify create_rag_prompt()
def create_rag_prompt(context: str, query: str) -> str:
    return f"""Your custom prompt: {context}
    Question: {query}"""
```

## 📚 API Reference

### RAGAgent Class

```python
class RAGAgent:
    def __init__(self, vector_store_manager=None)
    def initialize() -> bool
    def query(user_query: str) -> str
```

### Document Loader

```python
load_documents(folder: str) -> List[Document]
split_documents(docs: List[Document], chunk_size: int, overlap: int) -> List[Document]
```

### Vector Store

```python
class VectorStoreManager:
    def create_vector_store(documents) -> FAISS
    def load_vector_store() -> FAISS
    def get_or_create(documents) -> FAISS
    def retrieve_documents(query: str, k: int) -> List[Document]
```

### LLM

```python
generate_with_fallback(prompt: str, system_prompt: str = None) -> str
create_rag_prompt(context: str, query: str) -> str
```

### Ingestion

```python
ingest_documents(folder: str = None, force: bool = False) -> bool
reindex_documents(folder: str = None) -> bool
```

## 🐛 Troubleshooting

### Issue: "GOOGLE_API_KEY not found"
**Solution**: Create `.env` file and add your API key
```bash
cp .env.example .env
# Edit .env and add your key
```

### Issue: No documents loaded
**Solution**: Check document folder path and file formats
```bash
ls -la data/documents/
# Should show .pdf and .txt files
```

### Issue: Slow embeddings
**Solution**: Reduce chunk size or batch process
```env
CHUNK_SIZE=500  # Smaller chunks = faster
```

### Issue: LLM timeouts
**Solution**: Already handled! System auto-retries with next model
Check logs: `tail -f logs/rag_agent_*.log`

## 📝 License

This project is provided as-is for production use.

## 🤝 Contributing

To extend functionality:

1. Add feature to appropriate utility module
2. Update logging
3. Add error handling
4. Update `.env.example`
5. Document in README

## 📞 Support

For issues or questions:
- Check logs: `logs/rag_agent_*.log`
- Review configuration: `.env` file
- Run with verbose logging: `LOG_LEVEL=DEBUG`

---

**Built with:** LangChain • Google Gemini • FAISS • Python

**Last Updated:** March 18, 2024

**Version:** 1.0.0 - Production Ready ✅
