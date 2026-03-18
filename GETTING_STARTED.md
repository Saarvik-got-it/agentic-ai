# Getting Started with RAG Agent

## 5-Minute Quick Start

### Step 1: Clone and Setup
```bash
cd agentic-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure
```bash
# Copy env template
cp .env.example .env

# Edit .env and add your Google Gemini API key
# (Get key from https://aistudio.google.com/apikey)
```

### Step 3: Add Documents
```bash
# Copy your PDF/TXT files to the documents folder
cp your_file.pdf data/documents/
cp your_file.txt data/documents/
```

### Step 4: Verify Installation
```bash
python setup_verify.py
```

Expected output:
```
✅ All checks passed! Ready to use RAG Agent.
```

### Step 5: Start Using
```bash
# Interactive mode
python main.py

# Or ask a single question
python main.py -q "What is the main topic?"

# Or explicitly ingest documents
python main.py --ingest
```

## Common First Questions to Try

If you added the sample document, try:

```
Query: What are the main machine learning algorithms?
Query: What is supervised learning?
Query: What are the applications of machine learning?
Query: What challenges are mentioned?
```

## Next Steps

- **Read Full Documentation**: See README.md for complete guide
- **Add Your Documents**: Place PDF or TXT files in data/documents/
- **Integrate with FastAPI**: See README.md → Integration Examples
- **Customize Configuration**: Edit .env file to adjust behavior
- **Check Logs**: View logs/rag_agent_*.log for debugging

## Troubleshooting

### Import errors?
```bash
pip install -r requirements.txt --upgrade
python setup_verify.py
```

### API key issues?
```bash
# Check .env file
cat .env
# Make sure GOOGLE_API_KEY=xxx is set without "your_google_api_key_here"
```

### No documents found?
```bash
# Check documents folder
ls -la data/documents/
# Add PDF or TXT files there
```

### Getting errors in logs?
```bash
# Check recent logs
tail -f logs/rag_agent_*.log

# Enable debug mode
# Edit .env: LOG_LEVEL=DEBUG
# Then run again
```

## Architecture Overview

```
User Query
    ↓
RAG Agent
    ├─ Load Documents → Split into chunks
    ├─ Create Embeddings → Store in FAISS
    ├─ Retrieve similar chunks
    ├─ Generate response via Google Gemini
    └─ Return answer

All steps are logged and can be monitored!
```

## Key Files

- **main.py** - Entry point (CLI interface)
- **agents/rag_agent.py** - Main RAG logic
- **utils/** - Supporting utilities (embeddings, vector store, etc.)
- **data/documents/** - Your documents go here
- **.env** - Configuration (API keys, paths)
- **logs/rag_agent_*.log** - Debug logs

## Need Help?

1. Check README.md for detailed documentation
2. Review logs/rag_agent_*.log for error messages
3. Run setup_verify.py to diagnose issues
4. Ensure your .env file has GOOGLE_API_KEY set

---

Happy querying! 🚀
