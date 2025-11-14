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
import csv
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
        description="Run the RAG system on questions from QASPER, QMSum, or NarrativeQA datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=["qasper", "qmsum", "narrativeqa", "quality", "hotpot", "musique"],
        default="qasper",
        help="Dataset to use: 'qasper', 'qmsum', 'narrativeqa', 'quality', 'hotpot', or 'musique'.",
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


def iter_qasper_article_questions(dataset_split: Sequence[Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Iterate over QASPER dataset articles and questions."""
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


def load_narrativeqa_from_local(split: str, dataset_dir: Path) -> List[Dict[str, Any]]:
    """
    Load NarrativeQA dataset from local CSV files.
    
    Args:
        split: Dataset split ('train', 'val', 'validation', or 'test')
        dataset_dir: Path to NarrativeQA directory (e.g., Datasets/narrativeqa)
    
    Returns:
        List of story records as dictionaries with combined data from documents, qaps, and summaries
    """
    # Map split names (validation -> train set for NarrativeQA uses 'val' in summaries)
    split_map = {"validation": "train", "train": "train", "test": "test"}
    split_name = split_map.get(split.lower(), split.lower())
    
    # For validation, NarrativeQA might use 'val' in some files, but typically 'train' contains validation
    # We'll check the 'set' column in CSV files
    
    documents_csv = dataset_dir / "documents.csv"
    qaps_csv = dataset_dir / "qaps.csv"
    summaries_csv = dataset_dir / "third_party" / "wikipedia" / "summaries.csv"
    
    if not documents_csv.exists():
        raise FileNotFoundError(f"documents.csv not found in {dataset_dir}")
    if not qaps_csv.exists():
        raise FileNotFoundError(f"qaps.csv not found in {dataset_dir}")
    
    # Load documents
    documents = {}
    try:
        with documents_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_id = row.get("document_id")
                doc_set = row.get("set", "").lower()
                # Map validation to train for document loading
                if split.lower() == "validation" and doc_set == "train":
                    documents[doc_id] = row
                elif doc_set == split_name:
                    documents[doc_id] = row
    except Exception as exc:
        logging.error("Failed to load documents.csv: %s", exc)
        raise
    
    # Load QAs
    qas_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    try:
        with qaps_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_id = row.get("document_id")
                qa_set = row.get("set", "").lower()
                # Map validation to train for QA loading
                if split.lower() == "validation" and qa_set == "train":
                    if doc_id not in qas_by_doc:
                        qas_by_doc[doc_id] = []
                    qas_by_doc[doc_id].append(row)
                elif qa_set == split_name:
                    if doc_id not in qas_by_doc:
                        qas_by_doc[doc_id] = []
                    qas_by_doc[doc_id].append(row)
    except Exception as exc:
        logging.error("Failed to load qaps.csv: %s", exc)
        raise
    
    # Load summaries
    summaries = {}
    if summaries_csv.exists():
        try:
            with summaries_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    doc_id = row.get("document_id")
                    summary_set = row.get("set", "").lower()
                    # Map validation to train for summary loading
                    if split.lower() == "validation" and summary_set == "train":
                        summaries[doc_id] = row.get("summary", "")
                    elif summary_set == split_name:
                        summaries[doc_id] = row.get("summary", "")
        except Exception as exc:
            logging.warning("Failed to load summaries.csv: %s", exc)
    
    # Combine data: only include documents that have QAs
    records: List[Dict[str, Any]] = []
    for doc_id, doc_data in documents.items():
        if doc_id in qas_by_doc:
            # Create a record for each QA pair (NarrativeQA has multiple QAs per document)
            for qa in qas_by_doc[doc_id]:
                record = doc_data.copy()
                record["question"] = qa.get("question", "")
                # Combine answer1 and answer2 into a list
                answers = []
                if qa.get("answer1"):
                    answers.append(qa.get("answer1"))
                if qa.get("answer2") and qa.get("answer2") != qa.get("answer1"):
                    answers.append(qa.get("answer2"))
                record["answers"] = answers
                record["summary"] = summaries.get(doc_id, "")
                record["document_id"] = doc_id
                records.append(record)
    
    if not records:
        raise FileNotFoundError(
            f"No NarrativeQA records found for split '{split}'. "
            f"Found {len(documents)} documents and {len(qas_by_doc)} documents with QAs."
        )
    
    logging.info("Loaded %d NarrativeQA records from local CSV files", len(records))
    return records


def load_qmsum_from_local(split: str, dataset_dir: Path) -> List[Dict[str, Any]]:
    """
    Load QMSum dataset from local directory.
    
    Args:
        split: Dataset split ('train', 'val', 'validation', or 'test')
        dataset_dir: Path to QMSum data directory (e.g., Datasets/QMSum/data)
    
    Returns:
        List of meeting records as dictionaries
    """
    # Map split names (validation -> val)
    split_map = {"validation": "val", "train": "train", "test": "test"}
    split_dir = split_map.get(split.lower(), split.lower())
    
    # Try different domain directories (ALL is most comprehensive)
    domains = ["ALL", "Academic", "Product", "Committee"]
    records: List[Dict[str, Any]] = []
    
    for domain in domains:
        domain_path = dataset_dir / domain / split_dir
        if not domain_path.exists():
            logging.debug("Domain directory %s/%s does not exist, skipping", domain, split_dir)
            continue
        
        # Load all JSON files in the split directory
        json_files = list(domain_path.glob("*.json"))
        logging.info("Found %d JSON files in %s/%s", len(json_files), domain, split_dir)
        
        for json_file in json_files:
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    record = json.load(f)
                    # Add metadata fields
                    record["meeting_id"] = json_file.stem
                    record["domain"] = domain
                    record["split"] = split_dir
                    records.append(record)
            except Exception as exc:
                logging.warning("Failed to load %s: %s", json_file, exc)
    
    if not records:
        raise FileNotFoundError(
            f"No QMSum records found in {dataset_dir}. "
            f"Tried splits: {split_dir} in domains: {domains}"
        )
    
    logging.info("Loaded %d QMSum records from local directory", len(records))
    return records


def load_quality_from_local(split: str, dataset_dir: Path) -> List[Dict[str, Any]]:
    """
    Load QuALITY dataset from local JSONL files.
    
    Args:
        split: Dataset split ('train', 'val', 'validation', or 'test')
        dataset_dir: Path to QuALITY data directory (e.g., Datasets/quality/data)
    
    Returns:
        List of article records as dictionaries
    """
    # Map split names (validation -> dev)
    split_map = {"validation": "dev", "train": "train", "test": "test", "dev": "dev"}
    split_name = split_map.get(split.lower(), split.lower())
    
    # Try different version directories (v1.0.1 is latest)
    versions = ["v1.0.1", "v1.0", "v0.9"]
    records: List[Dict[str, Any]] = []
    
    for version in versions:
        version_dir = dataset_dir / version
        if not version_dir.exists():
            logging.debug("Version directory %s does not exist, skipping", version)
            continue
        
        # Try both regular and htmlstripped versions
        filename_patterns = [
            f"QuALITY.{version}.{split_name}",
            f"QuALITY.{version}.htmlstripped.{split_name}",
        ]
        
        for filename_pattern in filename_patterns:
            jsonl_file = version_dir / filename_pattern
            if jsonl_file.exists():
                logging.info("Found QuALITY file: %s", jsonl_file)
                try:
                    with jsonl_file.open("r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                                records.append(record)
                            except json.JSONDecodeError as exc:
                                logging.warning("Failed to parse line %d in %s: %s", line_num, jsonl_file, exc)
                    logging.info("Loaded %d records from %s", len(records), jsonl_file)
                    break  # Found and loaded from this version, no need to try other versions
                except Exception as exc:
                    logging.warning("Failed to load %s: %s", jsonl_file, exc)
        
        if records:
            break  # Successfully loaded, no need to try other versions
    
    if not records:
        raise FileNotFoundError(
            f"No QuALITY records found in {dataset_dir}. "
            f"Tried splits: {split_name} in versions: {versions}"
        )
    
    logging.info("Loaded %d QuALITY records from local directory", len(records))
    return records


def load_hotpot_from_local(split: str, dataset_dir: Path) -> List[Dict[str, Any]]:
    """
    Load HotpotQA dataset from local JSON files.
    
    Args:
        split: Dataset split ('train', 'val', 'validation', 'dev', or 'test')
        dataset_dir: Path to HotpotQA directory (e.g., Datasets/hotpot)
    
    Returns:
        List of question-answer records as dictionaries
    """
    # Map split names (validation -> dev)
    split_map = {"validation": "dev", "train": "train", "test": "test", "dev": "dev"}
    split_name = split_map.get(split.lower(), split.lower())
    
    # HotpotQA file naming patterns
    filename_patterns = [
        f"hotpot_{split_name}_distractor_v1.json",
        f"hotpot_{split_name}_fullwiki_v1.json",
        f"hotpot_{split_name}_v1.json",
        f"hotpot_{split_name}_v1.1.json",
    ]
    
    records: List[Dict[str, Any]] = []
    
    for filename_pattern in filename_patterns:
        json_file = dataset_dir / filename_pattern
        if json_file.exists():
            logging.info("Found HotpotQA file: %s", json_file)
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        records.extend(data)
                        logging.info("Loaded %d records from %s", len(data), json_file)
                        break  # Found and loaded, no need to try other patterns
                    else:
                        logging.warning("HotpotQA file %s does not contain a list", json_file)
            except Exception as exc:
                logging.warning("Failed to load %s: %s", json_file, exc)
    
    if not records:
        raise FileNotFoundError(
            f"No HotpotQA records found in {dataset_dir}. "
            f"Tried splits: {split_name} with patterns: {filename_patterns}"
        )
    
    logging.info("Loaded %d HotpotQA records from local directory", len(records))
    return records


def load_musique_from_local(split: str, dataset_dir: Path) -> List[Dict[str, Any]]:
    """
    Load MuSiQue dataset from local JSONL files.
    
    Args:
        split: Dataset split ('train', 'val', 'validation', 'dev', or 'test')
        dataset_dir: Path to MuSiQue directory (e.g., Datasets/musique)
    
    Returns:
        List of question-answer records as dictionaries
    """
    # Map split names (validation -> dev)
    split_map = {"validation": "dev", "train": "train", "test": "test", "dev": "dev"}
    split_name = split_map.get(split.lower(), split.lower())
    
    # MuSiQue data is typically in a data/ subdirectory
    data_dir = dataset_dir / "data"
    if not data_dir.exists():
        data_dir = dataset_dir  # Fallback to root directory
    
    records: List[Dict[str, Any]] = []
    
    # MuSiQue has two variants: ans (answerable) and full
    # Try both variants
    variants = ["ans", "full"]
    
    for variant in variants:
        filename_patterns = [
            f"musique_{variant}_v1.0_{split_name}.jsonl",
            f"musique_{variant}_v1.0_{split_name}.json",
        ]
        
        for filename_pattern in filename_patterns:
            jsonl_file = data_dir / filename_pattern
            if jsonl_file.exists():
                logging.info("Found MuSiQue file: %s", jsonl_file)
                try:
                    with jsonl_file.open("r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                                records.append(record)
                            except json.JSONDecodeError as exc:
                                logging.warning("Failed to parse line %d in %s: %s", line_num, jsonl_file, exc)
                    logging.info("Loaded %d records from %s", len(records), jsonl_file)
                    break  # Found and loaded, no need to try other variants
                except Exception as exc:
                    logging.warning("Failed to load %s: %s", jsonl_file, exc)
        
        if records:
            break  # Successfully loaded, no need to try other variants
    
    if not records:
        raise FileNotFoundError(
            f"No MuSiQue records found in {dataset_dir} or {data_dir}. "
            f"Tried splits: {split_name} with variants: {variants}"
        )
    
    logging.info("Loaded %d MuSiQue records from local directory", len(records))
    return records


def iter_qmsum_article_questions(dataset_split: Sequence[Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Iterate over QMSum dataset meetings and questions."""
    for example in dataset_split:
        # QMSum uses meeting_id or filename as identifier
        meeting_id = example.get("meeting_id") or example.get("id") or f"meeting-{hash(str(example))}"
        questions: List[Dict[str, Any]] = []
        
        # QMSum structure: has 'general_query_list' and 'specific_query_list'
        general_queries = example.get("general_query_list", [])
        specific_queries = example.get("specific_query_list", [])
        
        # Process general queries
        for query_item in general_queries:
            if isinstance(query_item, dict):
                query_text = query_item.get("query") or query_item.get("question", "")
                answer = query_item.get("answer") or ""
                if query_text:
                    questions.append({"question": query_text, "answers": [answer] if answer else []})
        
        # Process specific queries
        for query_item in specific_queries:
            if isinstance(query_item, dict):
                query_text = query_item.get("query") or query_item.get("question", "")
                answer = query_item.get("answer") or ""
                if query_text:
                    questions.append({"question": query_text, "answers": [answer] if answer else []})
        
        # Fallback: try alternative field names
        if not questions:
            question_list = example.get("question_list") or example.get("questions", [])
            answer_list = example.get("answer_list") or example.get("answers", [])
            
            if isinstance(question_list, list) and isinstance(answer_list, list):
                total = max(len(question_list), len(answer_list))
                for idx in range(total):
                    question_text = question_list[idx] if idx < len(question_list) else ""
                    answers = answer_list[idx] if idx < len(answer_list) else None
                    if isinstance(answers, str):
                        answers = [answers]
                    elif isinstance(answers, dict):
                        answer_text = answers.get("answer") or answers.get("text") or ""
                        if answer_text:
                            answers = [answer_text] if isinstance(answer_text, str) else answer_text
                        else:
                            answers = []
                    questions.append({"question": question_text, "answers": answers})
            elif isinstance(example.get("query"), str):
                answer = example.get("answer") or ""
                questions.append({"question": example.get("query", ""), "answers": [answer] if answer else []})

        if not questions:
            logging.debug("No questions found for meeting %s", meeting_id)
            continue
        
        yield meeting_id, example, questions


def iter_narrativeqa_article_questions(dataset_split: Sequence[Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Iterate over NarrativeQA dataset stories and questions."""
    # Group by document_id since CSV format creates one record per QA pair
    documents: Dict[str, Dict[str, Any]] = {}
    questions_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    
    for example in dataset_split:
        # NarrativeQA uses document_id as identifier
        story_id = example.get("document_id") or example.get("id") or example.get("example_id") or f"story-{hash(str(example))}"
        
        # Store document metadata (without question-specific fields)
        if story_id not in documents:
            doc_data = {k: v for k, v in example.items() 
                       if k not in ("question", "answers", "answer1", "answer2")}
            documents[story_id] = doc_data
        
        # Extract question and answers
        question_text = example.get("question") or example.get("question_text") or ""
        if question_text:
            answers = example.get("answers") or []
            if not answers:
                # Try answer1 and answer2 fields from CSV
                answer1 = example.get("answer1", "")
                answer2 = example.get("answer2", "")
                answers = []
                if answer1:
                    answers.append(answer1)
                if answer2 and answer2 != answer1:
                    answers.append(answer2)
            
            if story_id not in questions_by_doc:
                questions_by_doc[story_id] = []
            
            questions_by_doc[story_id].append({
                "question": question_text,
                "answers": answers if isinstance(answers, list) else [answers] if answers else []
            })
    
    # Yield one document with all its questions
    for story_id, doc_data in documents.items():
        questions = questions_by_doc.get(story_id, [])
        if not questions:
            logging.debug("No questions found for story %s", story_id)
            continue
        
        yield story_id, doc_data, questions


def iter_quality_article_questions(dataset_split: Sequence[Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Iterate over QuALITY dataset articles and questions."""
    for example in dataset_split:
        # QuALITY uses article_id as identifier
        article_id = example.get("article_id") or example.get("id") or f"article-{hash(str(example))}"
        questions: List[Dict[str, Any]] = []
        
        # QuALITY structure: has 'questions' list with multiple choice questions
        questions_list = example.get("questions", [])
        
        for q_item in questions_list:
            if isinstance(q_item, dict):
                question_text = q_item.get("question") or q_item.get("question_text") or ""
                if not question_text:
                    continue
                
                # QuALITY has multiple choice options
                options = q_item.get("options", [])
                gold_label = q_item.get("gold_label")
                
                # Extract the correct answer from options using gold_label (1-based index)
                answers = []
                if gold_label is not None and isinstance(options, list) and len(options) > 0:
                    # gold_label is 1-based index
                    try:
                        gold_idx = int(gold_label) - 1
                        if 0 <= gold_idx < len(options):
                            correct_answer = options[gold_idx]
                            answers.append(correct_answer)
                        else:
                            # If gold_label is invalid, try to use all options as reference
                            logging.warning("Invalid gold_label %d for question in article %s", gold_label, article_id)
                            answers = options[:1] if options else []
                    except (ValueError, TypeError):
                        logging.warning("Invalid gold_label type for question in article %s: %s", article_id, gold_label)
                        answers = options[:1] if options else []
                elif options:
                    # If no gold_label, use first option as fallback
                    answers = [options[0]] if options else []
                
                questions.append({"question": question_text, "answers": answers})
        
        if not questions:
            logging.debug("No questions found for article %s", article_id)
            continue
        
        yield article_id, example, questions


def iter_hotpot_article_questions(dataset_split: Sequence[Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Iterate over HotpotQA dataset questions."""
    # HotpotQA structure: each entry is a question-answer pair with context
    # We group by context (articles) when possible, but each entry can be standalone
    for idx, example in enumerate(dataset_split):
        # HotpotQA uses _id as identifier
        qa_id = example.get("_id") or example.get("id") or f"hotpot-{idx}"
        
        # HotpotQA has a single question per entry
        question_text = example.get("question") or ""
        if not question_text:
            logging.debug("No question found for HotpotQA entry %s", qa_id)
            continue
        
        # Extract answer
        answer = example.get("answer") or ""
        answers = [answer] if answer else []
        
        questions = [{"question": question_text, "answers": answers}]
        
        # Use qa_id as article_id since HotpotQA doesn't group by articles
        # The context will be built from the example's context field
        yield qa_id, example, questions


def iter_musique_article_questions(dataset_split: Sequence[Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Iterate over MuSiQue dataset questions."""
    # MuSiQue structure: each entry is a question-answer pair with paragraphs
    for idx, example in enumerate(dataset_split):
        # MuSiQue uses question_id or id as identifier
        qa_id = example.get("question_id") or example.get("id") or example.get("_id") or f"musique-{idx}"
        
        # MuSiQue has a single question per entry
        question_text = example.get("question") or ""
        if not question_text:
            logging.debug("No question found for MuSiQue entry %s", qa_id)
            continue
        
        # Extract answer(s) - MuSiQue may have answer or answers field
        answer = example.get("answer") or ""
        answers_list = example.get("answers") or []
        answers = []
        if answer:
            answers.append(answer)
        if isinstance(answers_list, list):
            answers.extend([str(a) for a in answers_list if a])
        elif isinstance(answers_list, str):
            if answers_list and answers_list not in answers:
                answers.append(answers_list)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_answers = []
        for a in answers:
            if a and a not in seen:
                seen.add(a)
                unique_answers.append(a)
        
        questions = [{"question": question_text, "answers": unique_answers}]
        
        # Use qa_id as article_id since MuSiQue doesn't group by articles
        # The context will be built from the example's paragraphs field
        yield qa_id, example, questions


def iter_article_questions(dataset_split: Sequence[Dict[str, Any]], dataset_name: str) -> Iterable[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Iterate over articles/meetings/stories and questions based on dataset type."""
    dataset_lower = dataset_name.lower()
    if dataset_lower == "qmsum":
        yield from iter_qmsum_article_questions(dataset_split)
    elif dataset_lower == "narrativeqa":
        yield from iter_narrativeqa_article_questions(dataset_split)
    elif dataset_lower == "quality":
        yield from iter_quality_article_questions(dataset_split)
    elif dataset_lower == "hotpot":
        yield from iter_hotpot_article_questions(dataset_split)
    elif dataset_lower == "musique":
        yield from iter_musique_article_questions(dataset_split)
    else:  # default to qasper
        yield from iter_qasper_article_questions(dataset_split)


def build_qmsum_documents(
    rag_system: RAGSystem,
    meeting_id: str,
    record: Dict[str, Any],
    chunk_size: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Build documents from QMSum meeting transcript."""
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
                    "article_id": meeting_id,
                    "section_title": section_title,
                    "paragraph_index": paragraph_index,
                    "chunk_offset": local_idx,
                    "text_preview": chunk[:200] + ("..." if len(chunk) > 200 else ""),
                    "title": record.get("meeting") or record.get("title") or f"Meeting {meeting_id}",
                }
            )

    # QMSum has 'meeting_transcripts' field with list of speaker/content dicts
    transcripts = record.get("meeting_transcripts") or record.get("transcript") or record.get("text")
    
    if isinstance(transcripts, list):
        # List of speaker/content dictionaries
        for idx, segment in enumerate(transcripts):
            if isinstance(segment, dict):
                segment_text = segment.get("content") or segment.get("text") or ""
                speaker = segment.get("speaker") or segment.get("name")
                section_title = f"Speaker: {speaker}" if speaker else None
                if segment_text:
                    add_text_block(segment_text, section_title, idx)
            elif isinstance(segment, str):
                add_text_block(segment, None, idx)
    elif isinstance(transcripts, str):
        # Single transcript string, split by sentences or paragraphs
        paragraphs = transcripts.split("\n\n") or [transcripts]
        for idx, para in enumerate(paragraphs):
            if para.strip():
                add_text_block(para.strip(), None, idx)
    
    # Also check for meeting_summary or summary as additional context
    summary = record.get("meeting_summary") or record.get("summary")
    if isinstance(summary, str) and summary.strip():
        add_text_block(summary.strip(), "Meeting Summary", 0)

    if not documents:
        logging.warning("No textual content extracted for meeting %s", meeting_id)

    return documents, doc_metadata


def build_narrativeqa_documents(
    rag_system: RAGSystem,
    story_id: str,
    record: Dict[str, Any],
    chunk_size: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Build documents from NarrativeQA story."""
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
                    "article_id": story_id,
                    "section_title": section_title,
                    "paragraph_index": paragraph_index,
                    "chunk_offset": local_idx,
                    "text_preview": chunk[:200] + ("..." if len(chunk) > 200 else ""),
                    "title": record.get("document_title") or record.get("title") or f"Story {story_id}",
                }
            )

    # NarrativeQA has 'document_text' or 'story' field with the full story
    # Note: CSV format may not include story text, only story_url
    story_text = record.get("document_text") or record.get("story") or record.get("text") or record.get("article")
    
    if isinstance(story_text, str) and story_text.strip():
        # Split by paragraphs (double newlines) or sentences
        paragraphs = story_text.split("\n\n")
        if len(paragraphs) == 1:
            # Try splitting by single newlines if no double newlines
            paragraphs = story_text.split("\n")
        
        for idx, para in enumerate(paragraphs):
            if para.strip():
                add_text_block(para.strip(), None, idx)
    elif isinstance(story_text, list):
        # List of paragraphs or sentences
        for idx, para in enumerate(story_text):
            if isinstance(para, str) and para.strip():
                add_text_block(para.strip(), None, idx)
            elif isinstance(para, dict):
                para_text = para.get("text") or para.get("content") or para.get("paragraph") or ""
                if para_text.strip():
                    section = para.get("section") or para.get("heading")
                    add_text_block(para_text.strip(), section, idx)
    else:
        # If no story text available, check if we have a story_url or wiki_title for metadata
        story_url = record.get("story_url") or ""
        wiki_title = record.get("wiki_title") or ""
        if story_url or wiki_title:
            logging.debug("Story %s has URL/title but no text content. URL: %s, Title: %s", story_id, story_url, wiki_title)
    
    # Include summary if available (this is critical for NarrativeQA when story text isn't available)
    summary = record.get("summary") or record.get("summary_text") or record.get("document_summary")
    if isinstance(summary, str) and summary.strip():
        add_text_block(summary.strip(), "Summary", 0)
    elif isinstance(summary, dict):
        summary_text = summary.get("text") or summary.get("summary") or ""
        if summary_text.strip():
            add_text_block(summary_text.strip(), "Summary", 0)
    
    # If still no documents, try to use wiki_title as a minimal document
    if not documents and record.get("wiki_title"):
        title_text = f"Title: {record.get('wiki_title')}"
        if record.get("wiki_url"):
            title_text += f"\nURL: {record.get('wiki_url')}"
        add_text_block(title_text, "Document Info", 0)

    if not documents:
        logging.warning("No textual content extracted for story %s", story_id)

    return documents, doc_metadata


def build_quality_documents(
    rag_system: RAGSystem,
    article_id: str,
    record: Dict[str, Any],
    chunk_size: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Build documents from QuALITY article (HTML content)."""
    import re
    from html.parser import HTMLParser
    
    documents: List[str] = []
    doc_metadata: List[Dict[str, Any]] = []
    
    class HTMLStripper(HTMLParser):
        """Simple HTML parser to extract text content."""
        def __init__(self):
            super().__init__()
            self.text = []
            self.skip = False
            
        def handle_starttag(self, tag, attrs):
            # Skip script and style tags
            if tag.lower() in ('script', 'style'):
                self.skip = True
                
        def handle_endtag(self, tag):
            if tag.lower() in ('script', 'style'):
                self.skip = False
            elif tag.lower() in ('p', 'div', 'br'):
                self.text.append('\n')
            elif tag.lower() in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                self.text.append('\n')
                
        def handle_data(self, data):
            if not self.skip:
                self.text.append(data)
    
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
                    "title": record.get("title") or f"Article {article_id}",
                }
            )
    
    # QuALITY has 'article' field with HTML content
    article_html = record.get("article") or record.get("html") or ""
    
    if article_html:
        # Strip HTML tags and extract text
        stripper = HTMLStripper()
        try:
            stripper.feed(article_html)
            article_text = ''.join(stripper.text)
            
            # Clean up the text (remove excessive whitespace)
            article_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', article_text)  # Multiple newlines to double
            article_text = re.sub(r' +', ' ', article_text)  # Multiple spaces to single
            
            # Split into paragraphs
            paragraphs = [p.strip() for p in article_text.split('\n\n') if p.strip()]
            
            current_section = None
            for idx, para in enumerate(paragraphs):
                if para.strip():
                    # Try to detect section headers (lines that are short and end without period)
                    if len(para) < 100 and not para.rstrip().endswith(('.', '!', '?', ':')) and not para.strip().startswith('<'):
                        current_section = para.strip()
                        continue
                    add_text_block(para.strip(), current_section, idx)
        except Exception as exc:
            logging.warning("Failed to parse HTML for article %s: %s. Using raw text.", article_id, exc)
            # Fallback: just extract text without HTML tags
            article_text = re.sub(r'<[^>]+>', ' ', article_html)
            article_text = re.sub(r'\s+', ' ', article_text)
            if article_text.strip():
                add_text_block(article_text.strip(), None, 0)
    
    if not documents:
        logging.warning("No textual content extracted for article %s", article_id)
    
    return documents, doc_metadata


def build_hotpot_documents(
    rag_system: RAGSystem,
    article_id: str,
    record: Dict[str, Any],
    chunk_size: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Build documents from HotpotQA context."""
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
                    "title": section_title or f"HotpotQA {article_id}",
                }
            )
    
    # HotpotQA has 'context' field with list of paragraphs
    # Each paragraph is [title, sentences] where sentences is a list of strings
    context = record.get("context") or []
    
    if isinstance(context, list):
        for para_idx, paragraph in enumerate(context):
            if isinstance(paragraph, list) and len(paragraph) >= 2:
                title = paragraph[0] if paragraph[0] else None
                sentences = paragraph[1] if isinstance(paragraph[1], list) else []
                if sentences:
                    para_text = " ".join(str(s) for s in sentences if s)
                    if para_text.strip():
                        add_text_block(para_text.strip(), title, para_idx)
            elif isinstance(paragraph, str):
                add_text_block(paragraph.strip(), None, para_idx)
    
    if not documents:
        logging.warning("No textual content extracted for HotpotQA entry %s", article_id)
    
    return documents, doc_metadata


def build_musique_documents(
    rag_system: RAGSystem,
    article_id: str,
    record: Dict[str, Any],
    chunk_size: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Build documents from MuSiQue paragraphs."""
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
                    "title": section_title or f"MuSiQue {article_id}",
                }
            )
    
    # MuSiQue has 'paragraphs' field with list of paragraph objects
    paragraphs = record.get("paragraphs") or []
    
    if isinstance(paragraphs, list):
        for para_idx, paragraph in enumerate(paragraphs):
            if isinstance(paragraph, dict):
                # Paragraph might have 'title' and 'text' or just 'text'
                title = paragraph.get("title") or paragraph.get("paragraph_title")
                para_text = paragraph.get("text") or paragraph.get("paragraph_text") or ""
                if para_text:
                    add_text_block(str(para_text).strip(), title, para_idx)
            elif isinstance(paragraph, str):
                add_text_block(paragraph.strip(), None, para_idx)
    
    if not documents:
        logging.warning("No textual content extracted for MuSiQue entry %s", article_id)
    
    return documents, doc_metadata


def build_article_documents(
    rag_system: RAGSystem,
    article_id: str,
    record: Dict[str, Any],
    chunk_size: int,
    dataset_name: str = "qasper",
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Build documents from article/meeting/story record based on dataset type."""
    dataset_lower = dataset_name.lower()
    if dataset_lower == "qmsum":
        return build_qmsum_documents(rag_system, article_id, record, chunk_size)
    elif dataset_lower == "narrativeqa":
        return build_narrativeqa_documents(rag_system, article_id, record, chunk_size)
    elif dataset_lower == "quality":
        return build_quality_documents(rag_system, article_id, record, chunk_size)
    elif dataset_lower == "hotpot":
        return build_hotpot_documents(rag_system, article_id, record, chunk_size)
    elif dataset_lower == "musique":
        return build_musique_documents(rag_system, article_id, record, chunk_size)
    
    # Default QASPER document building
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

    dataset_name = args.dataset.lower()
    if dataset_name == "narrativeqa":
        logger.info("Loading NarrativeQA dataset split '%s' from local directory...", args.split)
        # Try to load from local directory first
        narrativeqa_dir = CURRENT_DIR / "Datasets" / "narrativeqa"
        if narrativeqa_dir.exists():
            logger.info("Found NarrativeQA directory: %s", narrativeqa_dir)
            try:
                dataset_split = load_narrativeqa_from_local(args.split, narrativeqa_dir)
                logger.info("Loaded %d story records from local NarrativeQA directory", len(dataset_split))
            except Exception as exc:
                logger.error("Failed to load NarrativeQA from local directory: %s", exc)
                logger.info("Falling back to HuggingFace dataset...")
                try:
                    dataset_split = load_dataset("google-deepmind/narrativeqa", split=args.split)
                    logger.info("Loaded %d story records from HuggingFace NarrativeQA", len(dataset_split))
                except Exception as exc2:
                    logger.error("Failed to load NarrativeQA from HuggingFace: %s", exc2)
                    raise FileNotFoundError(
                        f"Could not load NarrativeQA dataset. Tried local directory {narrativeqa_dir} "
                        f"and HuggingFace. Error: {exc2}"
                    )
        else:
            logger.warning("Local NarrativeQA directory not found at %s, trying HuggingFace...", narrativeqa_dir)
            try:
                dataset_split = load_dataset("google-deepmind/narrativeqa", split=args.split)
                logger.info("Loaded %d story records from HuggingFace NarrativeQA", len(dataset_split))
            except Exception as exc:
                logger.error("Failed to load NarrativeQA from HuggingFace: %s", exc)
                raise FileNotFoundError(
                    f"Could not load NarrativeQA dataset. Local directory not found at {narrativeqa_dir} "
                    f"and HuggingFace load failed: {exc}"
                )
    elif dataset_name == "qmsum":
        logger.info("Loading QMSum dataset split '%s' from local directory...", args.split)
        # Try to load from local directory first
        qmsum_data_dir = CURRENT_DIR / "Datasets" / "QMSum" / "data"
        if qmsum_data_dir.exists():
            logger.info("Found QMSum data directory: %s", qmsum_data_dir)
            try:
                dataset_split = load_qmsum_from_local(args.split, qmsum_data_dir)
                logger.info("Loaded %d meeting records from local QMSum directory", len(dataset_split))
            except Exception as exc:
                logger.error("Failed to load QMSum from local directory: %s", exc)
                logger.info("Falling back to HuggingFace dataset...")
                try:
                    dataset_split = load_dataset("Yale-LILY/qmsum", split=args.split)
                    logger.info("Loaded %d meeting records from HuggingFace QMSum", len(dataset_split))
                except Exception as exc2:
                    logger.error("Failed to load QMSum from HuggingFace: %s", exc2)
                    try:
                        dataset_split = load_dataset("qmsum", split=args.split)
                        logger.info("Loaded %d meeting records from HuggingFace (alt name)", len(dataset_split))
                    except Exception as exc3:
                        logger.error("Failed to load QMSum with alternative name: %s", exc3)
                        raise FileNotFoundError(
                            f"Could not load QMSum dataset. Tried local directory {qmsum_data_dir} "
                            f"and HuggingFace. Error: {exc3}"
                        )
        else:
            logger.warning("Local QMSum directory not found at %s, trying HuggingFace...", qmsum_data_dir)
            try:
                dataset_split = load_dataset("Yale-LILY/qmsum", split=args.split)
                logger.info("Loaded %d meeting records from HuggingFace QMSum", len(dataset_split))
            except Exception as exc:
                logger.error("Failed to load QMSum from HuggingFace: %s", exc)
                raise FileNotFoundError(
                    f"Could not load QMSum dataset. Local directory not found at {qmsum_data_dir} "
                    f"and HuggingFace load failed: {exc}"
                )
    elif dataset_name == "quality":
        logger.info("Loading QuALITY dataset split '%s' from local directory...", args.split)
        quality_data_dir = CURRENT_DIR / "Datasets" / "quality" / "data"
        if quality_data_dir.exists():
            logger.info("Found QuALITY data directory: %s", quality_data_dir)
            try:
                dataset_split = load_quality_from_local(args.split, quality_data_dir)
                logger.info("Loaded %d article records from local QuALITY directory", len(dataset_split))
            except Exception as exc:
                logger.error("Failed to load QuALITY from local directory: %s", exc)
                raise FileNotFoundError(
                    f"Could not load QuALITY dataset from local directory. Error: {exc}"
                )
        else:
            raise FileNotFoundError(
                f"QuALITY data directory not found at {quality_data_dir}. "
                "Please ensure the dataset is available in the Datasets/quality/data directory."
            )
    elif dataset_name == "hotpot":
        logger.info("Loading HotpotQA dataset split '%s' from local directory...", args.split)
        hotpot_dir = CURRENT_DIR / "Datasets" / "hotpot"
        if hotpot_dir.exists():
            logger.info("Found HotpotQA directory: %s", hotpot_dir)
            try:
                dataset_split = load_hotpot_from_local(args.split, hotpot_dir)
                logger.info("Loaded %d question records from local HotpotQA directory", len(dataset_split))
            except Exception as exc:
                logger.error("Failed to load HotpotQA from local directory: %s", exc)
                logger.info("Falling back to HuggingFace dataset...")
                try:
                    dataset_split = load_dataset("hotpot_qa", split=args.split)
                    logger.info("Loaded %d question records from HuggingFace HotpotQA", len(dataset_split))
                except Exception as exc2:
                    logger.error("Failed to load HotpotQA from HuggingFace: %s", exc2)
                    raise FileNotFoundError(
                        f"Could not load HotpotQA dataset. Tried local directory {hotpot_dir} "
                        f"and HuggingFace. Error: {exc2}"
                    )
        else:
            logger.warning("Local HotpotQA directory not found at %s, trying HuggingFace...", hotpot_dir)
            try:
                dataset_split = load_dataset("hotpot_qa", split=args.split)
                logger.info("Loaded %d question records from HuggingFace HotpotQA", len(dataset_split))
            except Exception as exc:
                logger.error("Failed to load HotpotQA from HuggingFace: %s", exc)
                raise FileNotFoundError(
                    f"Could not load HotpotQA dataset. Local directory not found at {hotpot_dir} "
                    f"and HuggingFace load failed: {exc}"
                )
    elif dataset_name == "musique":
        logger.info("Loading MuSiQue dataset split '%s' from local directory...", args.split)
        musique_dir = CURRENT_DIR / "Datasets" / "musique"
        if musique_dir.exists():
            logger.info("Found MuSiQue directory: %s", musique_dir)
            try:
                dataset_split = load_musique_from_local(args.split, musique_dir)
                logger.info("Loaded %d question records from local MuSiQue directory", len(dataset_split))
            except Exception as exc:
                logger.error("Failed to load MuSiQue from local directory: %s", exc)
                raise FileNotFoundError(
                    f"Could not load MuSiQue dataset from local directory. Error: {exc}"
                )
        else:
            raise FileNotFoundError(
                f"MuSiQue directory not found at {musique_dir}. "
                "Please ensure the dataset is available in the Datasets/musique directory."
            )
    else:  # qasper
        logger.info("Loading QASPER dataset split '%s' ...", args.split)
        dataset_split = load_dataset("allenai/qasper", split=args.split)
        logger.info("Loaded %d question annotations from QASPER", len(dataset_split))

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
                output_path = candidate.resolve()
            try:
                assert output_path is not None
                logger.debug("Writing %d results to %s", len(collected_results), output_path)
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(collected_results, f, ensure_ascii=False, indent=2)
                logger.debug("Successfully wrote results to %s", output_path)
            except Exception as exc:
                logger.error("Failed to write results to %s: %s", output_path, exc)
                raise

    for article_id, record, questions in iter_article_questions(dataset_split, dataset_name):
        if args.articles and processed_articles >= args.articles:
            break

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
                    "dataset": dataset_name,
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

    # Final write to ensure all results are saved
    if args.save_json and collected_results:
        write_results()
        logger.info("Final write: Saved %d results to %s", len(collected_results), output_path)
    elif output_path:
        logger.info("Saved results to %s", output_path)


if __name__ == "__main__":
    main()


