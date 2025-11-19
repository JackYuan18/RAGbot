#!/usr/bin/env python3
"""
Utility script to exercise the RAG system against various datasets.

This script loads papers/articles/meetings from various datasets, indexes their
content with the existing RAG pipeline, and evaluates questions. It prints both
the generated answers and snippets of the retrieved context.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from datasets import load_dataset  # type: ignore
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'datasets'. Install it with `pip install datasets`."
    ) from exc

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Add RAGSystem to path
for candidate in (
    PROJECT_ROOT / "RAGSystem.py",
    PROJECT_ROOT / "RAG system" / "RAGSystem.py",
    PROJECT_ROOT / "RAGsystem" / "RAGSystem.py",
    CURRENT_DIR / "RAGSystem.py",
):
    if candidate.exists():
        module_dir = str(candidate.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        break

for extra_path in (PROJECT_ROOT / "NSTSCE", CURRENT_DIR):
    if extra_path.exists():
        path_str = str(extra_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from RAGSystem import RAGConfig, RAGSystem, setup_logging  # type: ignore

# Import our new modules
from dataset_loaders import (
    load_dataset_with_fallback,
    load_hotpot_from_local,
    load_longbench_from_local,
    load_musique_from_local,
    load_narrativeqa_from_local,
    load_qmsum_from_local,
    load_quality_from_local,
    load_wikiasp_from_local,
    load_xsum_from_local,
)
from dataset_iterators import iter_article_questions
from document_builders import build_article_documents
from evaluation_metrics import (
    calculate_bleu_score,
    calculate_exact_match,
    calculate_f1_score,
    calculate_recall_score,
    calculate_rouge_l_score,
    extract_reference_answers,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the RAG system on questions from various datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=["qasper", "qmsum", "narrativeqa", "quality", "hotpot", "musique", "xsum", "wikiasp", "longbench"],
        default="qasper",
        help="Dataset to use",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split to use (train, validation, test, or a slice).",
    )
    parser.add_argument(
        "--articles",
        type=int,
        default=1,
        help="Number of distinct papers to evaluate. Use 0 to process all papers in the split.",
    )
    parser.add_argument(
        "--questions-per-article",
        type=int,
        default=3,
        help="Maximum number of questions to test for each paper (0 means all questions).",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve for each query.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Maximum character length for each chunk fed to the retriever.",
    )
    parser.add_argument(
        "--generator-model",
        default="t5-small",
        help="Hugging Face model identifier used by the answer generator (or 'chatgpt5').",
    )
    parser.add_argument(
        "--chatgpt5-api-key",
        default=None,
        help="API key required when --generator-model chatgpt5 is selected.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--save-json",
        default="qasper_results.json",
        help="Optional path (relative to this script) to save results as JSON. Set to empty string to skip saving.",
    )
    parser.add_argument(
        "--show-context",
        dest="show_context",
        action="store_true",
        help="Display retrieved context in outputs.",
    )
    parser.add_argument(
        "--hide-context",
        dest="show_context",
        action="store_false",
        help="Hide retrieved context in outputs.",
    )
    parser.set_defaults(show_context=True)
    args = parser.parse_args()
    if args.generator_model.lower() == "chatgpt5" and not args.chatgpt5_api_key:
        parser.error("generator_model 'chatgpt5' requires --chatgpt5-api-key.")
    return args


def load_dataset_data(dataset_name: str, split: str) -> List[Dict[str, Any]]:
    """Load dataset based on name with appropriate loader."""
    dataset_name_lower = dataset_name.lower()
    
    dataset_configs = {
        "qasper": {
            "local_dir": None,
            "hf_name": "allenai/qasper",
            "loader": None,  # Use HuggingFace directly
        },
        "narrativeqa": {
            "local_dir": CURRENT_DIR / "Datasets" / "narrativeqa",
            "hf_name": "google-deepmind/narrativeqa",
            "loader": load_narrativeqa_from_local,
        },
        "qmsum": {
            "local_dir": CURRENT_DIR / "Datasets" / "QMSum" / "data",
            "hf_name": "Yale-LILY/qmsum",
            "loader": load_qmsum_from_local,
        },
        "quality": {
            "local_dir": CURRENT_DIR / "Datasets" / "quality" / "data",
            "hf_name": None,
            "loader": load_quality_from_local,
        },
        "hotpot": {
            "local_dir": CURRENT_DIR / "Datasets" / "hotpot",
            "hf_name": "hotpot_qa",
            "loader": load_hotpot_from_local,
        },
        "musique": {
            "local_dir": CURRENT_DIR / "Datasets" / "musique" / "data",
            "hf_name": None,
            "loader": load_musique_from_local,
        },
        "xsum": {
            "local_dir": CURRENT_DIR / "Datasets" / "xsum" ,
            "hf_name": "xsum",
            "loader": load_xsum_from_local,
        },
        "wikiasp": {
            "local_dir": CURRENT_DIR / "Datasets" / "wikiasp",
            "hf_name": None,
            "loader": load_wikiasp_from_local,
        },
        "longbench": {
            "local_dir": CURRENT_DIR / "Datasets" / "LongBench" / "data",
            "hf_name": "THUDM/LongBench",
            "loader": load_longbench_from_local,
        },
    }
    
    config = dataset_configs.get(dataset_name_lower)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # QASPER uses HuggingFace directly
    if dataset_name_lower == "qasper":
        logging.info("Loading QASPER dataset split '%s'...", split)
        dataset_split = load_dataset("allenai/qasper", split=split)
        return list(dataset_split)
    
    # Others use load_dataset_with_fallback
    return load_dataset_with_fallback(
        dataset_name=dataset_name,
        split=split,
        local_dir=config["local_dir"],
        hf_dataset_name=config["hf_name"],
        load_local_func=config["loader"],
    )


def print_results(
    article_id: str,
    title: str,
    question: str,
    answer: str,
    reference_answers: List[str],
    retrieved_metadata: List[Dict[str, Any]],
    exact_match: Optional[float],
    f1_score: Optional[float],
    recall_score: Optional[float],
    rouge_l_score: Optional[float],
    bleu_score: Optional[float],
    show_context: bool,
) -> None:
    """Print formatted results."""
    print("=" * 120)
    print(f"Article ID: {article_id}")
    print(f"Title: {title}")
    print(f"Question: {question}")
    print("\nGenerated Answer:\n")
    print(answer)
    
    if reference_answers:
        print("\nReference Answers:")
        for ref_idx, ref_answer in enumerate(reference_answers, start=1):
            print(f"  [{ref_idx}] {ref_answer}")
    
    print("\nScores:")
    if exact_match is not None:
        print(f"  Exact Match: {exact_match:.4f}")
    if f1_score is not None:
        print(f"  F1 Score: {f1_score:.4f}")
    if recall_score is not None:
        print(f"  Recall Score: {recall_score:.4f}")
    if rouge_l_score is not None:
        print(f"  ROUGE-L Score: {rouge_l_score:.4f}")
    if bleu_score is not None:
        print(f"  BLEU Score: {bleu_score:.4f}")
    
    if show_context and retrieved_metadata:
        print("\nRetrieved Context:")
        for meta_idx, meta in enumerate(retrieved_metadata, start=1):
            preview = meta.get("text_preview") or meta.get("chunk", "")[:200]
            section = meta.get("section_title") or "Unknown Section"
            paragraph_index = meta.get("paragraph_index")
            print(f"  ({meta_idx}) {section} ¶{paragraph_index}: {preview}")
    
    print("=" * 120)


def main() -> None:
    """Main execution function."""
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    use_chatgpt5 = args.generator_model.lower() == "chatgpt5"
    generator_model_name = "t5-small" if use_chatgpt5 else args.generator_model
    
    logger.info("Using %s generator", "ChatGPT5 via API" if use_chatgpt5 else f"model: {generator_model_name}")
    
    # Load dataset
    dataset_name = args.dataset.lower()
    logger.info("Processing dataset: %s (split: %s)", dataset_name, args.split)
    dataset_split = load_dataset_data(dataset_name, args.split)
    logger.info("Loaded %d records from %s dataset", len(dataset_split), dataset_name)
    
    # Initialize RAG system
    config = RAGConfig(
        chunk_size=args.chunk_size,
        retrieval_k=args.retrieval_k,
        generator_model=generator_model_name,
        use_chatgpt5=use_chatgpt5,
        openai_api_key=args.chatgpt5_api_key if use_chatgpt5 else None,
    )
    rag_system = RAGSystem(config)
    
    # Initialize results tracking
    processed_articles = 0
    total_questions = 0
    total_questions_in_dataset = 0  # Total questions in entire dataset (unfiltered)
    total_questions_available = 0  # Questions available for testing based on config limits
    collected_results: List[Dict[str, Any]] = []
    output_path: Optional[Path] = None
    
    def write_results() -> None:
        """Write results to JSON file."""
        if not args.save_json or not collected_results:
            return
        
        nonlocal output_path
        if output_path is None:
            candidate = Path(args.save_json)
            if not candidate.is_absolute():
                candidate = CURRENT_DIR / candidate
            candidate.parent.mkdir(parents=True, exist_ok=True)
            output_path = candidate.resolve()
        
        try:
            logger.debug("Writing %d results to %s (dataset: %s)", len(collected_results), output_path, dataset_name)
            # Ensure all results have the correct dataset name
            for result in collected_results:
                if "dataset" in result:
                    result["dataset"] = dataset_name
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(collected_results, f, ensure_ascii=False, indent=2)
            logger.debug("Successfully wrote results to %s", output_path)
        except Exception as exc:
            logger.error("Failed to write results to %s: %s", output_path, exc)
            raise
    
    # First pass: Count total questions in entire dataset (unfiltered) - no limits applied
    for article_id, record, questions in iter_article_questions(dataset_split, dataset_name):
        total_questions_in_dataset += len(questions)
    
    # Second pass: Count questions available for testing based on article and question limits
    article_count = 0
    for article_id, record, questions in iter_article_questions(dataset_split, dataset_name):
        # Count questions that would be tested based on article and question limits
        if args.articles == 0:  # If 0, process all articles
            # Process all articles
            if args.questions_per_article == 0:
                questions_limit = len(questions)  # All questions
            else:
                questions_limit = min(args.questions_per_article, len(questions))
        else:
            # Limited articles
            if article_count < args.articles:
                if args.questions_per_article == 0:
                    questions_limit = len(questions)  # All questions
                else:
                    questions_limit = min(args.questions_per_article, len(questions))
            else:
                questions_limit = 0
                break
        
        total_questions_available += questions_limit
        article_count += 1
        if args.articles and article_count >= args.articles:
            break
    
    # Process articles and questions
    processed_articles = 0
    for article_id, record, questions in iter_article_questions(dataset_split, dataset_name):
        if args.articles and processed_articles >= args.articles:
            break
        
        # Build documents
        documents, doc_metadata = build_article_documents(
            rag_system=rag_system,
            article_id=article_id,
            record=record,
            chunk_size=args.chunk_size,
            dataset_name=dataset_name,
        )
        
        if not documents:
            logger.warning("Skipping article %s because no documents were extracted", article_id)
            continue
        
        # Build index
        rag_system.retriever.build_index(documents, doc_metadata)
        logger.info(
            "Indexed %d chunks for article %s (%s)",
            len(documents),
            article_id,
            record.get("title", "unknown title"),
        )
        
        # Process questions
        questions_to_run = questions[:args.questions_per_article] if args.questions_per_article else questions
        questions_processed = 0
        
        for example in questions_to_run:
            question_text = (example.get("question") or "").strip()
            if not question_text:
                continue
            
            total_questions += 1
            questions_processed += 1
            
            logger.info("Q%d: %s", total_questions, question_text)
            
            # Query RAG system
            _, answer, retrieved_metadata = rag_system.query(question_text, k=args.retrieval_k)
            
            # Calculate metrics
            reference_answers = extract_reference_answers(example.get("answers"))
            exact_match = calculate_exact_match(answer, reference_answers)
            f1_score = calculate_f1_score(answer, reference_answers)
            rouge_l_score = calculate_rouge_l_score(answer, reference_answers)
            bleu_score = calculate_bleu_score(answer, reference_answers)
            
            # Calculate recall score
            recall_score = calculate_recall_score(answer, reference_answers)
            
            # Print results
            print_results(
                article_id=article_id,
                title=record.get("title") or "Unknown title",
                question=question_text,
                answer=answer,
                reference_answers=reference_answers,
                retrieved_metadata=retrieved_metadata,
                exact_match=exact_match,
                f1_score=f1_score,
                recall_score=recall_score,
                rouge_l_score=rouge_l_score,
                bleu_score=bleu_score,
                show_context=args.show_context,
            )
            
            # Store results - ensure dataset_name is used (not record.get("dataset") which might be sub-dataset name)
            result_entry = {
                "article_id": article_id,
                "title": record.get("title") or record.get("article_title") or "Unknown title",
                "question": question_text,
                "generated_answer": answer,
                "reference_answers": reference_answers,
                "retrieved_context": retrieved_metadata if args.show_context else [],
                "generator": "chatgpt5" if use_chatgpt5 else args.generator_model,
                "dataset": dataset_name,  # Use top-level dataset name, not sub-dataset from record
                "total_questions_in_dataset": total_questions_in_dataset,  # Total questions in entire dataset
                "total_questions_available": total_questions_available,  # Questions available based on config
                "bleu_score": bleu_score,
                "exact_match": exact_match,
                "f1_score": f1_score,
                "recall_score": recall_score,
                "rouge_l_score": rouge_l_score,
            }
            # For LongBench, also store the sub-dataset name for reference
            if dataset_name == "longbench" and record.get("dataset"):
                result_entry["sub_dataset"] = record.get("dataset")
            collected_results.append(result_entry)
            write_results()
        
        if questions_processed > 0:
            processed_articles += 1
    
    logger.info("Completed %d articles and processed %d questions.", processed_articles, total_questions)
    
    # Final write
    if args.save_json and collected_results:
        write_results()
        logger.info("Final write: Saved %d results to %s", len(collected_results), output_path)


if __name__ == "__main__":
    main()
