# Agentic AI: RAG + Content + Email Pipeline

This project is a modular, production-oriented multi-agent workflow for:

- Document-based question answering (RAG)
- Structured content transformation (Content Agent)
- Optional email delivery (Email Agent)
- Pipeline orchestration (simple orchestrator or LangGraph)

It is designed for both standalone CLI use and backend integration.

## What This Project Does

The system supports two primary operating modes:

1. RAG mode

- Loads PDF/TXT documents from a local folder
- Splits and embeds content with Google Gemini embeddings
- Stores vectors in FAISS for fast retrieval
- Answers user queries using retrieved context and Gemini LLM fallback

2. Pipeline mode

- Runs a multi-step flow: RAG -> Content -> Email (optional)
- Validates persona/content type from config
- Returns structured output and timing metrics
- Can run with either:
  - Standard orchestrator: pipelines/main_pipeline.py
  - LangGraph orchestrator: pipelines/langgraph_pipeline.py

## Core Features

- Config-driven behavior for personas and content types via config/content_agent_config.json
- Persistent FAISS vector store with automatic reuse
- Fallback model strategy for Gemini generation
- Optional SMTP email sending with Markdown-to-HTML rendering
- Interactive CLI and one-shot command mode
- Structured response objects with metrics and error handling

## APIs, Services, and Frameworks Used

### External APIs

- Google Gemini API (Generative Language API) for:
  - Embeddings via google.generativeai
  - Text generation/chat via LangChain's ChatGoogleGenerativeAI

### Services

- SMTP email service (Gmail-compatible by default) for outbound email delivery
- Local filesystem document store for source PDFs/TXT files
- Local FAISS index persistence under vector_store/faiss_index

### Frameworks and Libraries

- LangChain:
  - langchain (core orchestration utilities)
  - langchain-community (FAISS vector store, document loaders)
  - langchain-google-genai (Gemini chat model integration)
  - langchain-text-splitters (document chunking)
- LangGraph for graph-based pipeline orchestration (optional mode)
- FAISS (faiss-cpu) for local semantic vector retrieval
- Pydantic Settings (pydantic-settings) for typed environment-based config
- python-dotenv for .env loading
- PyPDF (via pypdf/PyPDFLoader) for PDF ingestion
- Markdown (python-markdown) for Markdown-to-HTML email rendering

### Runtime Interface

- CLI-first application using argparse
- No HTTP API is exposed in this repository currently (pipeline is structured to be backend/API-ready)

## Project Structure

```text
agentic-ai/
|-- agents/
|   |-- rag_agent.py
|   |-- content_agent.py
|   `-- email_agent.py
|-- pipelines/
|   |-- ingestion.py
|   |-- main_pipeline.py
|   |-- langgraph_pipeline.py
|   `-- pipeline_cli.py
|-- utils/
|   |-- config.py
|   |-- document_loader.py
|   |-- embeddings.py
|   |-- vector_store.py
|   |-- llm.py
|   `-- email_utils.py
|-- config/
|   `-- content_agent_config.json
|-- data/
|   `-- documents/
|-- vector_store/
|-- logs/
|-- main.py
|-- content_agent.py            # compatibility wrapper
|-- email_agent.py              # compatibility wrapper
|-- setup_verify.py
|-- test_pipeline.py
|-- requirements.txt
`-- .env.example
```

## Prerequisites

- Python 3.10+
- Google Gemini API key

## Installation

```bash
# from repository root
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
```

## Configuration

Create your environment file from the template:

```bash
copy .env.example .env
```

Set at least:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Optional email settings (required only when sending email):

```env
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_USE_TLS=true
```

## Add Documents

Place files under data/documents/.

Supported formats:

- .pdf
- .txt

## Quick Validation

Run setup checks:

```bash
python setup_verify.py
```

Run basic pipeline validation tests:

```bash
python test_pipeline.py
```

## Usage

### 1) RAG Agent (default behavior)

Interactive:

```bash
python main.py
```

Single query:

```bash
python main.py --query "What are the main topics?"
```

Explicit ingestion:

```bash
python main.py --ingest
```

### 2) Pipeline Mode (RAG -> Content -> Email optional)

Interactive pipeline:

```bash
python main.py --pipeline
```

Single run:

```bash
python main.py --pipeline --query "Explain machine learning"
```

Specify persona and content type:

```bash
python main.py --pipeline --query "Explain transformers" --persona technical_writer --content-type report
```

Skip RAG stage:

```bash
python main.py --pipeline --query "Use this text directly" --no-rag
```

List available options:

```bash
python main.py --pipeline --list-options
```

### 3) LangGraph Orchestration

Enable LangGraph orchestrator:

```bash
python main.py --pipeline --use-langgraph --query "Explain retrieval augmented generation"
```

The current graph flow is:

```text
START -> rag_node -> content_node -> email_node -> END
```

### 4) Send Final Output by Email

Pipeline output to email:

```bash
python main.py --pipeline --query "Weekly summary" --email recipient@example.com
```

With explicit subject:

```bash
python main.py --pipeline --query "Weekly summary" --email recipient@example.com --email-subject "Weekly AI Summary"
```

You can also force intent with:

```bash
python main.py --pipeline --query "Weekly summary" --send-email --email recipient@example.com
```

### 5) Standalone Content Agent

List options:

```bash
python content_agent.py --list-options
```

From inline text:

```bash
python content_agent.py --input-text "Factual input text" --content-type summary --persona research_analyst
```

From file:

```bash
python content_agent.py --input-file data\documents\sample_doc.txt --content-type report --persona technical_writer
```

### 6) Standalone Email Agent

```bash
python email_agent.py --input-text "Hello from the email agent" --to recipient@example.com
```

```bash
python email_agent.py --input-file output.txt --to recipient@example.com --subject "Custom subject"
```

## Available Personas and Content Types

Defined in config/content_agent_config.json.

Personas:

- technical_writer
- blog_writer
- research_analyst
- beginner_teacher

Content types:

- summary
- blog
- report
- explanation

## Environment Variables

Core:

- GOOGLE_API_KEY

RAG/vector settings:

- DOCUMENT_FOLDER (default: ./data/documents)
- VECTOR_STORE_PATH (default: ./vector_store/faiss_index)
- TOP_K_CHUNKS (default: 5)
- CHUNK_SIZE (default: 1000)
- CHUNK_OVERLAP (default: 200)

LLM and logging:

- LLM_TEMPERATURE (default: 0.3)
- LLM_MAX_TOKENS (default: 512)
- LOG_LEVEL (default: INFO)
- LOG_FILE (default: ./logs/rag_agent.log)

Email:

- EMAIL_HOST
- EMAIL_PORT
- EMAIL_USER
- EMAIL_PASSWORD
- EMAIL_USE_TLS

Content agent config override:

- CONTENT_AGENT_CONFIG (optional; default: ./config/content_agent_config.json)

## Pipeline Response Shape

Both orchestrators return a structured PipelineResponse containing:

- success
- query
- rag_output (None when --no-rag)
- final_output
- persona
- content_type
- email_status (when email stage runs)
- error
- metrics
- timestamp

Metrics include:

- total_duration
- validation_duration
- rag_duration
- content_duration
- email_duration

## Notes and Operational Behavior

- FAISS index is persisted and reused after first build.
- If no documents are present, RAG returns a clear guidance message.
- Pipeline mode is auto-detected when pipeline-specific flags are used.
- Email sending is blocking and depends on valid SMTP configuration.
- Markdown package is used for rich HTML emails, with graceful fallback when unavailable.

## Troubleshooting

1. Missing dependencies

```bash
pip install -r requirements.txt --upgrade
```

2. Invalid or missing API key

- Check .env
- Ensure GOOGLE_API_KEY is set and not placeholder text

3. No retrieval results

- Verify files exist in data/documents/
- Re-run ingestion with python main.py --ingest

4. Email sending fails

- Verify EMAIL_USER and EMAIL_PASSWORD
- For Gmail, use an app password (not account password)
- Confirm SMTP host/port and network access

## Typical End-to-End Flow

```text
User Query
   -> RAG retrieval from local documents
   -> Content transformation by persona/type
   -> Optional email delivery
   -> Structured response + metrics
```
