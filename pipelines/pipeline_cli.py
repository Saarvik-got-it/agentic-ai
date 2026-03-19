"""
Pipeline CLI Interface for multi-agent orchestration.

Provides interactive and command-line interfaces for running the main pipeline
that connects RAG Agent → Content Agent → Email Agent (optional).

Features:
- Interactive mode for easy experimentation
- Command-line flags for automation
- List available personas and content types
- Debug logging support
"""

import sys
import argparse
from typing import Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pipelines.main_pipeline import (
    run_pipeline,
    get_available_options,
    PipelineResponse,
)
from pipelines.langgraph_pipeline import run_langgraph_pipeline
from utils.logger import setup_logger

logger = setup_logger(__name__)


def display_response(response: PipelineResponse, verbose: bool = False) -> None:
    """
    Display pipeline response in user-friendly format.
    
    Args:
        response: PipelineResponse from pipeline execution
        verbose: Show detailed metrics and timing
    """
    print("\n" + "=" * 80)
    
    if response.success:
        print("✅ PIPELINE EXECUTED SUCCESSFULLY")
        print("=" * 80)
        
        print(f"\n📝 Query:")
        print(f"  {response.query[:100]}{'...' if len(response.query) > 100 else ''}")
        
        if response.rag_output:
            print(f"\n🔍 RAG Output:")
            print("-" * 80)
            print(response.rag_output[:500])
            if len(response.rag_output) > 500:
                print(f"... ({len(response.rag_output)} total characters)")
            print("-" * 80)
        
        print(f"\n✨ Final Output ({response.content_type} • {response.persona}):")
        print("-" * 80)
        print(response.final_output)
        print("-" * 80)

        if response.email_status:
            print("\n📧 Email Status:")
            print("-" * 80)
            print(f"Status: {response.email_status.get('status', 'unknown')}")
            if response.email_status.get("recipient"):
                print(f"Recipient: {response.email_status.get('recipient')}")
            if response.email_status.get("subject"):
                print(f"Subject: {response.email_status.get('subject')}")
            if response.email_status.get("error"):
                print(f"Error: {response.email_status.get('error')}")
            print("-" * 80)
        
        if verbose and response.metrics:
            print(f"\n⏱️  Metrics:")
            print(f"  Total Duration: {response.metrics.get('total_duration', 0):.2f}s")
            print(f"  Validation: {response.metrics.get('validation_duration', 0):.2f}s")
            print(f"  RAG Stage: {response.metrics.get('rag_duration', 0):.2f}s")
            print(f"  Content Stage: {response.metrics.get('content_duration', 0):.2f}s")
            print(f"  Email Stage: {response.metrics.get('email_duration', 0):.2f}s")
        
        if response.timestamp:
            print(f"  Generated: {response.timestamp}")
    
    else:
        print("❌ PIPELINE EXECUTION FAILED")
        print("=" * 80)
        print(f"\n⚠️  Error: {response.error}")
        
        if verbose and response.metrics:
            print(f"\n⏱️  Partial Metrics:")
            print(f"  Total Duration: {response.metrics.get('total_duration', 0):.2f}s")
    
    print("\n" + "=" * 80 + "\n")


def interactive_mode(use_langgraph: bool = False) -> None:
    """Run pipeline in interactive mode."""
    print("\n" + "=" * 80)
    print("🔗 PIPELINE - Multi-Agent Orchestration")
    print("=" * 80)
    print("Connect RAG Agent → Content Agent → Email Agent (optional)\n")
    print(
        "Orchestration Engine: "
        + ("LangGraph" if use_langgraph else "Simple pipeline")
        + "\n"
    )
    
    # Load and display available options
    try:
        options = get_available_options()
        print("Available Personas:")
        for p in options["personas"]:
            print(f"  • {p}")
        
        print("\nAvailable Content Types:")
        for ct in options["content_types"]:
            print(f"  • {ct}")
        
        print("\nType 'exit' to quit. Press Ctrl+C to interrupt.\n")
    
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return
    
    try:
        while True:
            # Query
            query = input("\n📝 Enter query (or type 'exit'): ").strip()
            
            if query.lower() in ["exit", "quit", "bye"]:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                print("⚠️  Query cannot be empty.")
                continue
            
            # Content Type
            content_type = input(
                f"📋 Content type [{options['content_types'][0]}]: "
            ).strip()
            if not content_type:
                content_type = options["content_types"][0]
            
            # Persona
            persona = input(
                f"👤 Persona [{options['personas'][0]}]: "
            ).strip()
            if not persona:
                persona = options["personas"][0]
            
            # Use RAG
            rag_choice = input("Use RAG retrieval? [Y/n]: ").strip().lower()
            use_rag = rag_choice != "n"

            # Optional email
            email = input("📧 Send output via email? Enter recipient (or leave empty): ").strip()
            email_subject = None
            if email:
                email_subject = input("✉️  Email subject [auto]: ").strip() or None
            
            # Execute pipeline
            print("\n⏳ Processing...")
            runner = run_langgraph_pipeline if use_langgraph else run_pipeline
            response = runner(
                query=query,
                content_type=content_type,
                persona=persona,
                use_rag=use_rag,
                debug=False,
                send_email=bool(email),
                email=email or None,
                email_subject=email_subject,
            )
            
            display_response(response, verbose=False)
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


def cli_single_execution(
    query: str,
    content_type: Optional[str] = None,
    persona: Optional[str] = None,
    no_rag: bool = False,
    debug: bool = False,
    verbose: bool = False,
    use_langgraph: bool = False,
    send_email: bool = False,
    email: Optional[str] = None,
    email_subject: Optional[str] = None,
) -> None:
    """
    Execute pipeline from command-line arguments.
    
    Args:
        query: User query
        content_type: Content type (uses default if not specified)
        persona: Persona (uses default if not specified)
        no_rag: Skip RAG stage if True
        debug: Enable debug logging
        verbose: Show detailed output
        use_langgraph: Use LangGraph orchestration instead of simple pipeline
        send_email: Whether to send final output via email
        email: Recipient email address
        email_subject: Optional custom subject line
    """
    # Load defaults if not specified
    try:
        options = get_available_options()
        
        if not persona:
            persona = options["personas"][0]
        if not content_type:
            content_type = options["content_types"][0]
        
        # Validate
        if persona not in options["personas"]:
            print(f"❌ Invalid persona: {persona}")
            return
        if content_type not in options["content_types"]:
            print(f"❌ Invalid content type: {content_type}")
            return
    
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # Execute
    print("\n⏳ Processing...")
    runner = run_langgraph_pipeline if use_langgraph else run_pipeline
    response = runner(
        query=query,
        content_type=content_type,
        persona=persona,
        use_rag=not no_rag,
        debug=debug,
        send_email=send_email or bool(email),
        email=email,
        email_subject=email_subject,
    )
    
    display_response(response, verbose=verbose)


def list_options() -> None:
    """Print available personas and content types."""
    try:
        options = get_available_options()
        
        print("\n" + "=" * 80)
        print("AVAILABLE PIPELINE OPTIONS")
        print("=" * 80)
        
        print("\nPersonas:")
        for p in options["personas"]:
            print(f"  • {p}")
        
        print("\nContent Types:")
        for ct in options["content_types"]:
            print(f"  • {ct}")
        
        print("\n" + "=" * 80 + "\n")
    
    except Exception as e:
        print(f"❌ Failed to load options: {e}")


def main():
    """Main CLI entry point for pipeline."""
    parser = argparse.ArgumentParser(
        description="Pipeline - Multi-Agent Orchestration (RAG → Content)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

  Interactive mode:
    python main.py --pipeline

  Single query with defaults:
    python main.py --pipeline --query "Explain machine learning"

  With specific persona and content type:
    python main.py --pipeline --query "Query text" \\
                   --persona technical_writer --content-type blog

  Skip RAG stage:
    python main.py --pipeline --query "Already formatted text" --no-rag

  List available options:
    python main.py --pipeline --list-options

  Debug mode:
    python main.py --pipeline --query "Query" --debug

    LangGraph orchestration:
        python main.py --pipeline --use-langgraph --query "Query"

    Send final output via email:
        python main.py --pipeline --query "Query" --email user@example.com

DEFAULTS:
  • Persona: technical_writer
  • Content Type: summary
  • RAG: enabled
    • Orchestration: simple pipeline (use --use-langgraph to switch)

PERSONAS & CONTENT TYPES:
  Use --list-options to see available personas and content types.
        """
    )
    
    # Pipeline flag
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run the pipeline orchestrator (RAG → Content)"
    )
    
    # Query
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Query text (runs in non-interactive mode)"
    )
    
    # Persona
    parser.add_argument(
        "--persona",
        type=str,
        help="Target persona for content generation"
    )
    
    # Content type
    parser.add_argument(
        "--content-type",
        type=str,
        help="Type of content to generate"
    )
    
    # RAG control
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Skip RAG retrieval stage (use query directly)"
    )
    
    # List options
    parser.add_argument(
        "--list-options",
        action="store_true",
        help="List available personas and content types"
    )
    
    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--use-langgraph",
        action="store_true",
        help="Use LangGraph orchestration engine"
    )

    parser.add_argument(
        "--send-email",
        action="store_true",
        help="Send final output via Email Agent"
    )

    parser.add_argument(
        "--email",
        type=str,
        help="Recipient email address for Email Agent"
    )

    parser.add_argument(
        "--email-subject",
        type=str,
        help="Optional email subject line"
    )
    
    # Verbose
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output including metrics"
    )
    
    args = parser.parse_args()
    
    # Must have --pipeline flag
    if not args.pipeline:
        return
    
    # List options
    if args.list_options:
        list_options()
    
    # Single query
    elif args.query:
        cli_single_execution(
            query=args.query,
            content_type=args.content_type,
            persona=args.persona,
            no_rag=args.no_rag,
            debug=args.debug,
            verbose=args.verbose,
            use_langgraph=args.use_langgraph,
            send_email=args.send_email,
            email=args.email,
            email_subject=args.email_subject,
        )
    
    # Interactive mode (default)
    else:
        interactive_mode(use_langgraph=args.use_langgraph)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)
