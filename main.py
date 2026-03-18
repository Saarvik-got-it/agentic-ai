"""
CLI Interface for RAG Agent and Pipeline Orchestrator.
Allows interactive querying, document ingestion, and multi-agent pipeline orchestration.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.rag_agent import RAGAgent
from pipelines.ingestion import ingest_documents
from utils.config import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


def main_pipeline():
    """Route to pipeline CLI."""
    from pipelines.pipeline_cli import main as pipeline_main
    pipeline_main()


def main_interactive():
    """Run RAG Agent in interactive mode."""
    print("\n" + "="*60)
    print("RAG Agent - Interactive Mode")
    print("="*60)
    print("Type your questions below. Press Ctrl+C to exit.\n")
    
    # Initialize agent
    agent = RAGAgent()
    
    try:
        while True:
            user_input = input("\n📝 Query: ").strip()
            
            if not user_input:
                print("Please enter a query.")
                continue
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\n👋 Goodbye!")
                break
            
            print("\n🔍 Processing your query...")
            response = agent.query(user_input)
            
            print("\n📄 Answer:")
            print("-" * 60)
            print(response)
            print("-" * 60)
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


def main_ingest():
    """Run document ingestion pipeline."""
    print("\n" + "="*60)
    print("Document Ingestion Pipeline")
    print("="*60)
    
    settings = get_settings()
    print(f"\nIngesting documents from: {settings.document_folder}")
    
    success = ingest_documents()
    
    if success:
        print("\n✅ Documents ingested successfully!")
    else:
        print("\n❌ Document ingestion failed. Check logs for details.")
        sys.exit(1)


def main_query(query_text: str):
    """Run single query."""
    agent = RAGAgent()
    response = agent.query(query_text)
    print(f"\n🔍 Query: {query_text}")
    print("\n📄 Answer:")
    print("-" * 60)
    print(response)
    print("-" * 60)


def main():
    """Main CLI entry point for RAG Agent."""
    parser = argparse.ArgumentParser(
        description="RAG Agent & Pipeline - Document-based QA and Multi-Agent Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

  RAG Agent (default):
    python main.py                    # Interactive RAG mode
    python main.py -q "Your question"  # Single RAG query
    python main.py --ingest            # Ingest documents

  Pipeline (multi-agent orchestration):
    python main.py --pipeline          # Interactive pipeline mode
    python main.py --pipeline --query "Text" --persona technical_writer
    python main.py --pipeline --list-options

  Help:
    python main.py --help              # Show this help
        """
    )
    
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run multi-agent pipeline (RAG → Content) instead of RAG agent"
    )
    
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Run a single query and exit"
    )
    
    parser.add_argument(
        "-i", "--ingest",
        action="store_true",
        help="Run document ingestion pipeline"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Pipeline-specific arguments (when --pipeline is used)
    parser.add_argument(
        "--persona",
        type=str,
        help="[Pipeline] Target persona for content generation"
    )
    
    parser.add_argument(
        "--content-type",
        type=str,
        help="[Pipeline] Type of content to generate"
    )
    
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="[Pipeline] Skip RAG retrieval stage"
    )
    
    parser.add_argument(
        "--list-options",
        action="store_true",
        help="[Pipeline] List available personas and content types"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Auto-detect pipeline mode if pipeline-specific arguments are provided
    is_pipeline_mode = (
        args.pipeline or 
        args.persona is not None or 
        args.content_type is not None or 
        args.no_rag or 
        args.list_options
    )
    
    # Pipeline mode (explicit or auto-detected)
    if is_pipeline_mode:
        # Import and run pipeline CLI
        from pipelines.pipeline_cli import (
            interactive_mode,
            cli_single_execution,
            list_options as pipeline_list_options,
        )
        
        if args.list_options:
            pipeline_list_options()
        elif args.query:
            cli_single_execution(
                query=args.query,
                content_type=args.content_type,
                persona=args.persona,
                no_rag=args.no_rag,
                debug=args.debug,
                verbose=args.verbose
            )
        else:
            interactive_mode()
    
    # Ingest documents (RAG mode)
    elif args.ingest:
        main_ingest()
    
    # Single query (RAG mode)
    elif args.query:
        main_query(args.query)
    
    # Interactive RAG mode (default)
    else:
        main_interactive()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)
