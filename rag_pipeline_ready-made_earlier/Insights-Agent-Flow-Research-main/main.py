"""
Main entry point for the AI Leadership Insight Agent.

Supports three modes:
  --ingest           Run the document ingestion pipeline
  --question "..."   Ask a single question
  --interactive      Start an interactive chat session
  --stream           Stream pipeline steps in real time
"""

import os
import sys
import json
import subprocess
import argparse
import logging

# Fix Windows console encoding for special characters
if sys.platform == "win32":
    subprocess.run(["chcp", "65001"], shell=True, capture_output=True)
    os.environ["PYTHONIOENCODING"] = "utf-8"

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_ingest(company: str = None, force: bool = False):
    """Run the document ingestion pipeline."""
    if company:
        import config
        config.set_active_company(company)
    from ingestion.ingest import ingest_pipeline

    print(f"\nStarting document ingestion for '{config._ACTIVE_COMPANY}'...")
    result = ingest_pipeline(force=force)

    if result.get("skipped"):
        print(f"\n⏭️  Ingestion Skipped: {result.get('reason', 'No changes detected')}")
        return result

    print("\nIngestion Complete!")
    print(f"   Documents: {result.get('num_documents', 0)}")
    print(f"   Files:     {result.get('num_files', 0)}")
    print(f"   Chunks:    {result.get('num_chunks', 0)}")
    print(f"   Parents:   {result.get('num_parents', 0)}")

    return result


def run_question(question: str, stream: bool = False):
    """Run a single question through the agent."""
    from graph.workflow import run_agent, stream_agent

    print(f"\nQuestion: {question}")
    print("=" * 60)

    if stream:
        print("\nProcessing...\n")
        final_chart = {}
        for node_name, state_update in stream_agent(question):
            print(f"  -> {node_name}")
            if "generation" in state_update and state_update["generation"]:
                print(f"\nAnswer:\n{state_update['generation']}")
            if "chart_data" in state_update and state_update["chart_data"]:
                final_chart = state_update["chart_data"]
        if final_chart:
            _print_chart_info(final_chart)
    else:
        result = run_agent(question)
        print(f"\nAnswer:\n{result.get('generation', 'No answer generated')}")

        # Show chart data
        if result.get("chart_data"):
            _print_chart_info(result["chart_data"])

        # Show metadata for debugging
        if result.get("route"):
            print(f"\nRoute: {result['route']}")
        if result.get("rewritten_question"):
            print(f"Rewritten: {result['rewritten_question'][:100]}...")
        if result.get("query_metadata"):
            print(f"Metadata: {json.dumps(result['query_metadata'], indent=2)}")
        if result.get("reranked_documents"):
            print(f"Documents used: {len(result['reranked_documents'])}")


def _print_chart_info(chart_data: dict):
    """Display chart data in the terminal."""
    print(f"\nChart Available: {chart_data.get('title', 'Untitled')}")
    print(f"   Type: {chart_data.get('chart_type', 'bar')}")
    print(f"   Data Points: {len(chart_data.get('labels', []))}")
    for label, value in zip(chart_data.get('labels', []), chart_data.get('values', [])):
        unit = chart_data.get('unit', '')
        max_val = max(chart_data.get('values', [1]))
        bar_len = max(1, int(value / max_val * 30)) if max_val > 0 else 1
        bar = '#' * bar_len
        print(f"   {label:25s} {bar} {value} {unit}")
    print("   Tip: Run 'streamlit run app.py' for interactive charts")


def run_interactive():
    """Run in interactive chat mode."""
    from graph.workflow import run_agent

    print("\nAI Leadership Insight Agent")
    print("=" * 60)
    print("Ask questions about your company documents.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("\nGoodbye!")
                break

            result = run_agent(question)
            answer = result.get('generation', 'No answer generated')
            print(f"\nAgent: {answer}")

            # Show chart data if available
            if result.get("chart_data"):
                _print_chart_info(result["chart_data"])

            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\nError processing question: {e}\n")


def main():
    import config
    parser = argparse.ArgumentParser(description="AI Leadership Insight Agent")
    parser.add_argument("--ingest", action="store_true", help="Run document ingestion")
    parser.add_argument("--question", type=str, help="Ask a single question")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--stream", action="store_true", help="Stream pipeline steps")
    parser.add_argument("--force", action="store_true",
                        help="Force re-ingestion even if files haven't changed")
    parser.add_argument("--company", type=str, default="infosys",
                        choices=list(config.AVAILABLE_COMPANIES.keys()),
                        help=f"Company to query (default: infosys). "
                             f"Available: {list(config.AVAILABLE_COMPANIES.keys())}. "
                             f"Add a new folder named <company>_company_docs to register a new company.")

    args = parser.parse_args()

    # Set active company before any operation
    config.set_active_company(args.company)
    print(f"Active company: {args.company}")

    if args.ingest:
        run_ingest(args.company, force=args.force)
    elif args.question:
        run_question(args.question, stream=args.stream)
    elif args.interactive:
        run_interactive()
    else:
        # Default: ingest then interactive
        print("No arguments provided. Running ingestion then interactive mode.")
        print("Use --help to see all options.\n")
        run_ingest(args.company)
        run_interactive()


if __name__ == "__main__":
    main()
