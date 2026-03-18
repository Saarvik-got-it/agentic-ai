"""
CLI Interface for RAG Agent.
Allows interactive querying and batch processing.
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
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Agent - Document-based Question Answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Interactive mode
  python main.py -q "Your question"  # Single query
  python main.py --ingest            # Ingest documents
  python main.py --help              # Show this help
        """
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
    
    args = parser.parse_args()
    
    # Ingest documents
    if args.ingest:
        main_ingest()
    
    # Single query
    elif args.query:
        main_query(args.query)
    
    # Interactive mode (default)
    else:
        main_interactive()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)
