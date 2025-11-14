#!/usr/bin/env python3
"""
Utility script to exercise the RAG system against the allenai/qasper dataset.

This script loads a configurable number of papers from the QASPER validation (or
other) split, indexes their paragraphs with the existing RAG pipeline, and
evaluates a handful of questions per paper. It prints both the generated
answers and snippets of the retrieved context so you can manually inspect the
system behaviour.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from datasets import load_dataset  # type: ignore
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'datasets'. Install it with `pip install datasets`."
    ) from exc

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore
    import nltk  # type: ignore
    
    # Download necessary NLTK data if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ModuleNotFoundError:
    sentence_bleu = None  # type: ignore
    SmoothingFunction = None  # type: ignore
    logging.warning("NLTK not available. BLEU scores will not be calculated.")

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

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

for extra_path in (
    PROJECT_ROOT / "NSTSCE",
    CURRENT_DIR,
):
    if extra_path.exists():
        path_str = str(extra_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from RAGSystem import RAGConfig, RAGSystem, setup_logging  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RAG system on questions from allenai/qasper."
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


def extract_reference_answers(answers: Any) -> List[str]:
    collected: List[str] = []

    def add_text(text: str) -> None:
        clean = text.strip()
        if clean:
            collected.append(clean)

    if isinstance(answers, dict):
        if "answer" in answers and isinstance(answers["answer"], list):
            for annotation in answers["answer"]:
                if not isinstance(annotation, dict):
                    continue
                if annotation.get("free_form_answer"):
                    add_text(annotation["free_form_answer"])
                for span in annotation.get("extractive_spans", []) or []:
                    add_text(span)
                if not annotation.get("free_form_answer") and not annotation.get("extractive_spans"):
                    for evidence in annotation.get("evidence", []) or []:
                        add_text(evidence)
        else:
            for value in answers.values():
                collected.extend(extract_reference_answers(value))
    elif isinstance(answers, list):
        for item in answers:
            collected.extend(extract_reference_answers(item))
    elif isinstance(answers, str):
        add_text(answers)

    seen = set()
    unique: List[str] = []
    for text in collected:
        if text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, and remove extra whitespace."""
    return " ".join(text.lower().strip().split())


def calculate_exact_match(generated_answer: str, reference_answers: List[str]) -> Optional[float]:
    """
    Calculate Exact Match score between generated answer and reference answers.
    
    Returns 1.0 if the generated answer exactly matches any reference answer (after normalization),
    or 0.0 if no match is found. Returns None if there are no reference answers.
    """
    if not reference_answers:
        return None
    
    try:
        normalized_generated = normalize_text(generated_answer)
        
        if not normalized_generated:
            return 0.0
        
        for ref_answer in reference_answers:
            if not ref_answer.strip():
                continue
            normalized_ref = normalize_text(ref_answer)
            if normalized_generated == normalized_ref:
                return 1.0
        
        return 0.0
    except Exception as exc:
        logging.warning("Failed to calculate Exact Match score: %s", exc)
        return None


def calculate_f1_score(generated_answer: str, reference_answers: List[str]) -> Optional[float]:
    """
    Calculate F1 score (token-level) between generated answer and reference answers.
    
    Returns the maximum F1 score across all reference answers, or None if
    there are no reference answers.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    where precision = common_tokens / generated_tokens
    and recall = common_tokens / reference_tokens
    """
    if not reference_answers:
        return None
    
    try:
        generated_tokens = set(normalize_text(generated_answer).split())
        
        if not generated_tokens:
            return 0.0
        
        max_f1 = 0.0
        
        for ref_answer in reference_answers:
            if not ref_answer.strip():
                continue
            reference_tokens = set(normalize_text(ref_answer).split())
            if not reference_tokens:
                continue
            
            # Calculate common tokens
            common_tokens = generated_tokens & reference_tokens
            
            if not common_tokens:
                continue
            
            # Calculate precision and recall
            precision = len(common_tokens) / len(generated_tokens) if generated_tokens else 0.0
            recall = len(common_tokens) / len(reference_tokens) if reference_tokens else 0.0
            
            # Calculate F1 score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                max_f1 = max(max_f1, f1)
        
        return max_f1 if max_f1 > 0.0 else None
    except Exception as exc:
        logging.warning("Failed to calculate F1 score: %s", exc)
        return None


def longest_common_subsequence_length(seq1: List[str], seq2: List[str]) -> int:
    """
    Calculate the length of the longest common subsequence (LCS) between two sequences.
    
    Uses dynamic programming approach.
    """
    if not seq1 or not seq2:
        return 0
    
    m, n = len(seq1), len(seq2)
    # Create a 2D table to store LCS lengths
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the table bottom-up
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def calculate_rouge_l_score(generated_answer: str, reference_answers: List[str], beta: float = 1.2) -> Optional[float]:
    """
    Calculate ROUGE-L score between generated answer and reference answers.
    
    ROUGE-L measures the longest common subsequence (LCS) based F-measure.
    
    Formula:
    - R_LCS = LCS_length / reference_length (recall)
    - P_LCS = LCS_length / candidate_length (precision)
    - F_LCS = ((1 + beta^2) * R_LCS * P_LCS) / (R_LCS + beta^2 * P_LCS)
    
    Returns the maximum ROUGE-L score across all reference answers, or None if
    there are no reference answers.
    
    Args:
        generated_answer: The generated answer text
        reference_answers: List of reference answer texts
        beta: Parameter for F-measure calculation (default 1.2, common for ROUGE-L)
    """
    if not reference_answers:
        return None
    
    try:
        generated_tokens = normalize_text(generated_answer).split()
        
        if not generated_tokens:
            return 0.0
        
        max_rouge_l = 0.0
        
        for ref_answer in reference_answers:
            if not ref_answer.strip():
                continue
            reference_tokens = normalize_text(ref_answer).split()
            if not reference_tokens:
                continue
            
            # Calculate LCS length
            lcs_length = longest_common_subsequence_length(reference_tokens, generated_tokens)
            
            if lcs_length == 0:
                continue
            
            # Calculate recall and precision
            recall_lcs = lcs_length / len(reference_tokens) if reference_tokens else 0.0
            precision_lcs = lcs_length / len(generated_tokens) if generated_tokens else 0.0
            
            # Calculate ROUGE-L F-score
            if recall_lcs + precision_lcs > 0:
                rouge_l = (
                    (1 + beta * beta) * recall_lcs * precision_lcs
                ) / (recall_lcs + beta * beta * precision_lcs)
                max_rouge_l = max(max_rouge_l, rouge_l)
        
        return max_rouge_l if max_rouge_l > 0.0 else None
    except Exception as exc:
        logging.warning("Failed to calculate ROUGE-L score: %s", exc)
        return None


def calculate_bleu_score(generated_answer: str, reference_answers: List[str]) -> Optional[float]:
    """
    Calculate BLEU score between generated answer and reference answers.
    
    Returns the maximum BLEU score across all reference answers, or None if
    NLTK is not available or if there are no reference answers.
    """
    if sentence_bleu is None or not reference_answers:
        return None
    
    try:
        # Tokenize generated answer
        generated_tokens = generated_answer.lower().split()
        
        if not generated_tokens:
            return 0.0
        
        # Calculate BLEU score for each reference answer and take the maximum
        smoothing = SmoothingFunction().method1
        max_bleu = 0.0
        
        for ref_answer in reference_answers:
            if not ref_answer.strip():
                continue
            reference_tokens = ref_answer.lower().split()
            if not reference_tokens:
                continue
            
            # Calculate BLEU score with smoothing to handle cases where n-grams don't match
            bleu = sentence_bleu(
                [reference_tokens],
                generated_tokens,
                smoothing_function=smoothing
            )
            max_bleu = max(max_bleu, bleu)
        
        return max_bleu if max_bleu > 0.0 else None
    except Exception as exc:
        logging.warning("Failed to calculate BLEU score: %s", exc)
        return None


def iter_article_questions(dataset_split: Sequence[Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    for example in dataset_split:
        article_id = example.get("article_id") or example.get("id") or example.get("title") or "unknown-article"
        questions: List[Dict[str, Any]] = []
        qas = example.get("qas")
        if isinstance(qas, dict) and isinstance(qas.get("question"), list):
            questions_list = qas.get("question", [])
            answers_list = qas.get("answers", [])
            total = max(len(questions_list), len(answers_list))
            for idx in range(total):
                question_text = questions_list[idx] if idx < len(questions_list) else ""
                answers = answers_list[idx] if idx < len(answers_list) else None
                questions.append({"question": question_text, "answers": answers})
        elif isinstance(example.get("question"), str):
            questions.append({"question": example.get("question", ""), "answers": example.get("answers")})
        else:
            logging.debug("No questions found for article %s", article_id)

        if not questions:
            continue
        yield article_id, example, questions


def build_article_documents(
    rag_system: RAGSystem,
    article_id: str,
    record: Dict[str, Any],
    chunk_size: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    documents: List[str] = []
    doc_metadata: List[Dict[str, Any]] = []

    def add_text_block(text: str, section_title: str | None, paragraph_index: int) -> None:
        if not text:
            return
        prefix = f"{section_title}\n" if section_title else ""
        chunk_source = prefix + text.strip()
        if not chunk_source:
            return
        chunks = (
            rag_system.pdf_processor._chunk_text(chunk_source)  # pylint: disable=protected-access
            if len(chunk_source) > chunk_size
            else [chunk_source]
        )
        for local_idx, chunk in enumerate(chunks):
            documents.append(chunk)
            doc_metadata.append(
                {
                    "article_id": article_id,
                    "section_title": section_title,
                    "paragraph_index": paragraph_index,
                    "chunk_offset": local_idx,
                    "text_preview": chunk[:200] + ("..." if len(chunk) > 200 else ""),
                    "title": record.get("title"),
                }
            )

    abstract = record.get("abstract")
    if isinstance(abstract, str):
        add_text_block(abstract, "Abstract", 0)
    elif isinstance(abstract, list):
        for idx, paragraph in enumerate(abstract):
            add_text_block(paragraph, "Abstract", idx)

    metadata = record.get("metadata")
    full_text = None
    if isinstance(metadata, dict):
        full_text = metadata.get("full_text") or metadata.get("fullText")

    if full_text is None:
        full_text = record.get("full_text") or record.get("fullText")

    if isinstance(full_text, dict):
        section_names = full_text.get("section_name") or full_text.get("sectionTitle") or full_text.get("titles") or []
        paragraphs = full_text.get("paragraphs") or full_text.get("text") or []
        for section_index, section_paragraphs in enumerate(paragraphs):
            if not isinstance(section_paragraphs, (list, tuple)):
                continue
            section_title = None
            if isinstance(section_names, (list, tuple)) and section_index < len(section_names):
                section_title = section_names[section_index]
            for paragraph_index, paragraph in enumerate(section_paragraphs):
                add_text_block(paragraph, section_title or f"Section {section_index + 1}", paragraph_index)
    elif isinstance(full_text, list):
        for section_index, section in enumerate(full_text):
            section_title = None
            section_paragraphs: Iterable[str] = []
            if isinstance(section, dict):
                section_title = (
                    section.get("section_title")
                    or section.get("sectionTitle")
                    or section.get("title")
                    or section.get("heading")
                )
                section_paragraphs = section.get("paragraphs") or section.get("text") or []
            elif isinstance(section, str):
                section_paragraphs = [section]
            elif isinstance(section, (list, tuple)):
                section_paragraphs = section

            for paragraph_index, paragraph in enumerate(section_paragraphs):
                add_text_block(paragraph, section_title or f"Section {section_index + 1}", paragraph_index)

    if not documents:
        logging.warning("No textual content extracted for article %s", article_id)

    return documents, doc_metadata


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    use_chatgpt5 = args.generator_model.lower() == "chatgpt5"
    generator_model_name = args.generator_model
    if use_chatgpt5:
        generator_model_name = "t5-small"
        logger.info("Using ChatGPT5 generator via API")
    else:
        logger.info("Using generator model: %s", generator_model_name)

    logger.info("Loading QASPER split '%s' ...", args.split)
    dataset_split = load_dataset("allenai/qasper", split=args.split)
    logger.info("Loaded %d question annotations", len(dataset_split))

    config = RAGConfig(
        chunk_size=args.chunk_size,
        retrieval_k=args.retrieval_k,
        generator_model=generator_model_name,
        use_chatgpt5=use_chatgpt5,
        openai_api_key=args.chatgpt5_api_key if use_chatgpt5 else None,
    )
    rag_system = RAGSystem(config)

    processed_articles = 0
    total_questions = 0
    collected_results: List[Dict[str, Any]] = []

    output_path: Optional[Path] = None

    def write_results() -> None:
        if args.save_json:
            nonlocal output_path
            if output_path is None:
                candidate = Path(args.save_json)
                if not candidate.is_absolute():
                    candidate = CURRENT_DIR / candidate
                candidate.parent.mkdir(parents=True, exist_ok=True)
                output_path = candidate
            try:
                assert output_path is not None
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(collected_results, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.error("Failed to write results to %s: %s", output_path, exc)

    for article_id, record, questions in iter_article_questions(dataset_split):
        if args.articles and processed_articles >= args.articles:
            break

        documents, doc_metadata = build_article_documents(
            rag_system=rag_system,
            article_id=article_id,
            record=record,
            chunk_size=args.chunk_size,
        )

        if not documents:
            logger.warning("Skipping article %s because no documents were extracted", article_id)
            continue

        rag_system.retriever.build_index(documents, doc_metadata)
        logger.info(
            "Indexed %d chunks for article %s (%s)",
            len(documents),
            article_id,
            record.get("title", "unknown title"),
        )

        questions_to_run = questions
        if args.questions_per_article:
            questions_to_run = questions[: args.questions_per_article]

        questions_processed = 0

        for example in questions_to_run:
            question_text = (example.get("question") or "").strip()
            if not question_text:
                continue
            total_questions += 1
            questions_processed += 1

            logger.info("Q%d: %s", total_questions, question_text)
            _, answer, retrieved_metadata = rag_system.query(question_text, k=args.retrieval_k)
            reference_answers = extract_reference_answers(example.get("answers"))
            bleu_score = calculate_bleu_score(answer, reference_answers)
            exact_match = calculate_exact_match(answer, reference_answers)
            f1_score = calculate_f1_score(answer, reference_answers)
            rouge_l_score = calculate_rouge_l_score(answer, reference_answers)

            print("=" * 120)
            print(f"Article ID: {article_id}")
            print(f"Title: {record.get('title') or 'Unknown title'}")
            print(f"Question: {question_text}")
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
            if rouge_l_score is not None:
                print(f"  ROUGE-L Score: {rouge_l_score:.4f}")
            if bleu_score is not None:
                print(f"  BLEU Score: {bleu_score:.4f}")

            if args.show_context and retrieved_metadata:
                print("\nRetrieved Context:")
                for meta_idx, meta in enumerate(retrieved_metadata, start=1):
                    preview = meta.get("text_preview") or meta.get("chunk", "")[:200]
                    section = meta.get("section_title") or "Unknown Section"
                    paragraph_index = meta.get("paragraph_index")
                    print(f"  ({meta_idx}) {section} ¶{paragraph_index}: {preview}")

            print("=" * 120)

            collected_results.append(
                {
                    "article_id": article_id,
                    "title": record.get("title") or "Unknown title",
                    "question": question_text,
                    "generated_answer": answer,
                    "reference_answers": reference_answers,
                    "retrieved_context": retrieved_metadata if args.show_context else [],
                    "generator": "chatgpt5" if use_chatgpt5 else args.generator_model,
                    "bleu_score": bleu_score,
                    "exact_match": exact_match,
                    "f1_score": f1_score,
                    "rouge_l_score": rouge_l_score,
                }
            )
            write_results()

        if questions_processed > 0:
            processed_articles += 1

    logger.info(
        "Completed %d articles and processed %d questions.", processed_articles, total_questions
    )

    if output_path:
        logger.info("Saved results to %s", output_path)


if __name__ == "__main__":
    main()


