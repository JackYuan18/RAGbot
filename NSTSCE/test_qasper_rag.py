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
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    from datasets import load_dataset  # type: ignore
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'datasets'. Install it with `pip install datasets`."
    ) from exc

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

for candidate in (
    PROJECT_ROOT / "RAGSystem.py",
    PROJECT_ROOT / "RAG system" / "RAGSystem.py",
    CURRENT_DIR / "RAGSystem.py",
):
    if candidate.exists():
        module_dir = str(candidate.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        break

from RAGSystem import RAGConfig, RAGSystem, setup_logging


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
        help="Hugging Face model identifier used by the answer generator.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


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

    logger.info("Loading QASPER split '%s' ...", args.split)
    dataset_split = load_dataset("allenai/qasper", split=args.split)
    logger.info("Loaded %d question annotations", len(dataset_split))

    config = RAGConfig(
        chunk_size=args.chunk_size,
        retrieval_k=args.retrieval_k,
        generator_model=args.generator_model,
    )
    rag_system = RAGSystem(config)

    processed_articles = 0
    total_questions = 0

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

            if retrieved_metadata:
                print("\nRetrieved Context:")
                for meta_idx, meta in enumerate(retrieved_metadata, start=1):
                    preview = meta.get("text_preview") or meta.get("chunk", "")[:200]
                    section = meta.get("section_title") or "Unknown Section"
                    paragraph_index = meta.get("paragraph_index")
                    print(f"  ({meta_idx}) {section} ¶{paragraph_index}: {preview}")

            print("=" * 120)

        if questions_processed > 0:
            processed_articles += 1

    logger.info(
        "Completed %d articles and processed %d questions.", processed_articles, total_questions
    )


if __name__ == "__main__":
    main()


