"""
RAG Evaluation Script — Measures pipeline quality using Ragas metrics.

Runs the agent pipeline against a golden QA dataset and computes:
  - Faithfulness:       Is the answer grounded in retrieved context?
  - Answer Relevancy:   How relevant is the answer to the question?
  - Context Precision:  Are the right documents ranked highly?
  - Context Recall:     Are all necessary docs retrieved?

Usage:
    python evaluation/evaluate.py --company infosys
    python evaluation/evaluate.py --company infosys --output evaluation/results.json
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from graph.workflow import run_agent

# Ragas imports
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_golden_dataset(path: str) -> list[dict]:
    """Load the golden QA dataset from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline_for_evaluation(questions: list[dict]) -> list[dict]:
    """
    Run all questions through the agent and collect answers + retrieved contexts.
    
    Returns list of dicts: {question, answer, contexts, ground_truth}
    """
    results = []
    total = len(questions)
    
    for i, item in enumerate(questions, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        logger.info(f"[{i}/{total}] Running query: {question[:80]}...")
        start = time.time()
        
        try:
            result = run_agent(question)
            answer = result.get("generation", "")
            
            # Extract retrieved contexts from the reranked documents
            contexts = []
            for doc in result.get("reranked_documents", []):
                if isinstance(doc, dict):
                    text = doc.get("text", doc.get("content", ""))
                    if text:
                        contexts.append(text)
                elif hasattr(doc, "page_content"):
                    contexts.append(doc.page_content)
            
            # If no reranked docs, try the raw documents
            if not contexts:
                for doc in result.get("documents", []):
                    if isinstance(doc, dict):
                        text = doc.get("text", doc.get("content", ""))
                        if text:
                            contexts.append(text)
            
            elapsed = time.time() - start
            logger.info(f"  Answer length: {len(answer)} chars | "
                       f"Contexts: {len(contexts)} | Time: {elapsed:.1f}s")
            
            results.append({
                "question": question,
                "answer": answer,
                "contexts": contexts if contexts else ["No context retrieved"],
                "ground_truth": ground_truth,
                "elapsed_seconds": round(elapsed, 2),
            })
            
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results.append({
                "question": question,
                "answer": f"ERROR: {e}",
                "contexts": ["Error during retrieval"],
                "ground_truth": ground_truth,
                "elapsed_seconds": 0,
            })
    
    return results


def run_ragas_evaluation(pipeline_results: list[dict]) -> dict:
    """
    Run Ragas evaluation on pipeline results.
    
    Returns dict with per-question scores and overall averages.
    """
    # Build Ragas EvaluationDataset
    samples = []
    for r in pipeline_results:
        sample = SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        )
        samples.append(sample)
    
    eval_dataset = EvaluationDataset(samples=samples)
    
    # Configure Ragas LLM and Embeddings
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=0,
    ))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
    ))
    
    # Define metrics
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
    ]
    
    logger.info("Running Ragas evaluation (this may take a few minutes)...")
    ragas_result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
    )
    
    return ragas_result


def format_results(pipeline_results: list[dict], ragas_result) -> dict:
    """Format results into a comprehensive report dict."""
    # Convert Ragas result to a DataFrame for per-question scores
    df = ragas_result.to_pandas()
    logger.info(f"Ragas Result Columns: {df.columns.tolist()}")
    
    per_question = []
    for i, r in enumerate(pipeline_results):
        row = {
            "question": r["question"],
            "answer_preview": r["answer"][:200] + "..." if len(r["answer"]) > 200 else r["answer"],
            "num_contexts": len(r["contexts"]),
            "elapsed_seconds": r["elapsed_seconds"],
            "faithfulness": round(float(df.iloc[i].get("faithfulness", 0.0)), 4),
            "answer_relevancy": round(float(df.iloc[i].get("answer_relevancy", 0.0)), 4),
            "context_precision": round(float(df.iloc[i].get("context_precision", 0.0)), 4),
            "context_recall": round(float(df.iloc[i].get("context_recall", 0.0)), 4),
        }
        # Fallback for different column names in newer Ragas versions
        if "context_precision" not in df.columns and "llm_context_precision_with_reference" in df.columns:
             row["context_precision"] = round(float(df.iloc[i].get("llm_context_precision_with_reference", 0.0)), 4)
             
        per_question.append(row)
    
    # Compute overall averages safely
    avg = {}
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "llm_context_precision_with_reference"]:
        if metric in df.columns:
            avg[metric] = round(df[metric].mean(), 4)
    
    # Normalize keys for report
    if "llm_context_precision_with_reference" in avg:
        avg["context_precision"] = avg.pop("llm_context_precision_with_reference")
        
    # Ensure all keys exist
    for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if k not in avg:
            avg[k] = 0.0
    
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "company": config._ACTIVE_COMPANY,
            "model": config.LLM_MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
            "num_questions": len(pipeline_results),
        },
        "overall_scores": avg,
        "per_question_scores": per_question,
    }
    
    return report


def print_results_table(report: dict):
    """Pretty-print the evaluation results to console."""
    print("\n" + "=" * 80)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 80)
    print(f"  Company:    {report['metadata']['company']}")
    print(f"  Model:      {report['metadata']['model']}")
    print(f"  Questions:  {report['metadata']['num_questions']}")
    print(f"  Timestamp:  {report['metadata']['timestamp']}")
    print("-" * 80)
    
    # Overall scores
    print("\n  OVERALL SCORES:")
    avg = report["overall_scores"]
    print(f"    Faithfulness:       {avg['faithfulness']:.4f}")
    print(f"    Answer Relevancy:   {avg['answer_relevancy']:.4f}")
    print(f"    Context Precision:  {avg['context_precision']:.4f}")
    print(f"    Context Recall:     {avg['context_recall']:.4f}")
    
    # Per-question breakdown
    print("\n  PER-QUESTION BREAKDOWN:")
    print("-" * 80)
    for i, q in enumerate(report["per_question_scores"], 1):
        print(f"\n  Q{i}: {q['question']}")
        print(f"      Faith: {q['faithfulness']:.4f} | "
              f"Relev: {q['answer_relevancy']:.4f} | "
              f"Prec:  {q['context_precision']:.4f} | "
              f"Recall: {q['context_recall']:.4f} | "
              f"Time: {q['elapsed_seconds']:.1f}s")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run Ragas evaluation on the RAG pipeline")
    parser.add_argument("--company", type=str, default="infosys",
                        choices=list(config.AVAILABLE_COMPANIES.keys()),
                        help="Company to evaluate")
    parser.add_argument("--dataset", type=str,
                        default=os.path.join(os.path.dirname(__file__), "golden_dataset.json"),
                        help="Path to golden dataset JSON")
    parser.add_argument("--output", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results.json"),
                        help="Path to save results JSON")
    
    args = parser.parse_args()
    
    # Set active company
    config.set_active_company(args.company)
    logger.info(f"Evaluating pipeline for company: {args.company}")
    
    # Step 1: Load golden dataset
    golden_data = load_golden_dataset(args.dataset)
    logger.info(f"Loaded {len(golden_data)} questions from {args.dataset}")
    
    # Step 2: Run pipeline on all questions
    logger.info("Running agent pipeline on all questions...")
    pipeline_results = run_pipeline_for_evaluation(golden_data)
    
    # Step 3: Run Ragas evaluation
    ragas_result = run_ragas_evaluation(pipeline_results)
    
    # Step 4: Format and save results
    report = format_results(pipeline_results, ragas_result)
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Results saved to {args.output}")
    
    # Step 5: Print results
    print_results_table(report)
    
    return report


if __name__ == "__main__":
    main()
