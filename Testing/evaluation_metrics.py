"""Evaluation metrics for RAG system testing."""

from __future__ import annotations

import logging
from typing import List, Optional

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


def extract_reference_answers(answers: any) -> List[str]:
    """Extract reference answers from various answer formats."""
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


def calculate_recall_score(generated_answer: str, reference_answers: List[str]) -> Optional[float]:
    """
    Calculate Recall score (token-level) between generated answer and reference answers.
    
    Returns the maximum recall score across all reference answers, or None if
    there are no reference answers.
    
    Recall = common_tokens / reference_tokens
    This measures how many tokens from the reference answer are present in the generated answer.
    """
    if not reference_answers:
        return None
    
    try:
        generated_tokens = set(normalize_text(generated_answer).split())
        if not generated_tokens:
            return 0.0
        
        max_recall = 0.0
        for ref_answer in reference_answers:
            if not ref_answer.strip():
                continue
            reference_tokens = set(normalize_text(ref_answer).split())
            if not reference_tokens:
                continue
            
            common_tokens = generated_tokens & reference_tokens
            recall = len(common_tokens) / len(reference_tokens) if reference_tokens else 0.0
            max_recall = max(max_recall, recall)
        
        return max_recall if max_recall > 0.0 else 0.0
    except Exception as exc:
        logging.warning("Failed to calculate Recall score: %s", exc)
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
            
            common_tokens = generated_tokens & reference_tokens
            if not common_tokens:
                continue
            
            precision = len(common_tokens) / len(generated_tokens) if generated_tokens else 0.0
            recall = len(common_tokens) / len(reference_tokens) if reference_tokens else 0.0
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                max_f1 = max(max_f1, f1)
        
        return max_f1 if max_f1 > 0.0 else None
    except Exception as exc:
        logging.warning("Failed to calculate F1 score: %s", exc)
        return None


def longest_common_subsequence_length(seq1: List[str], seq2: List[str]) -> int:
    """Calculate the length of the longest common subsequence (LCS) between two sequences."""
    if not seq1 or not seq2:
        return 0
    
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
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
    Returns the maximum ROUGE-L score across all reference answers, or None if
    there are no reference answers.
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
            
            lcs_length = longest_common_subsequence_length(reference_tokens, generated_tokens)
            if lcs_length == 0:
                continue
            
            recall_lcs = lcs_length / len(reference_tokens) if reference_tokens else 0.0
            precision_lcs = lcs_length / len(generated_tokens) if generated_tokens else 0.0
            
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
        generated_tokens = generated_answer.lower().split()
        if not generated_tokens:
            return 0.0
        
        smoothing = SmoothingFunction().method1
        max_bleu = 0.0
        
        for ref_answer in reference_answers:
            if not ref_answer.strip():
                continue
            reference_tokens = ref_answer.lower().split()
            if not reference_tokens:
                continue
            
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

