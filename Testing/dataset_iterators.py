"""Dataset iterator functions for extracting questions from various datasets."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Sequence, Tuple


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


def iter_qmsum_article_questions(dataset_split: Sequence[Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Iterate over QMSum dataset meetings and questions."""
    for example in dataset_split:
        meeting_id = example.get("meeting_id") or example.get("id") or f"meeting-{hash(str(example))}"
        questions: List[Dict[str, Any]] = []
        
        # Process general and specific queries
        for query_list_name in ["general_query_list", "specific_query_list"]:
            queries = example.get(query_list_name, [])
            for query_item in queries:
                if isinstance(query_item, dict):
                    query_text = query_item.get("query") or query_item.get("question", "")
                    answer = query_item.get("answer") or ""
                    if query_text:
                        questions.append({"question": query_text, "answers": [answer] if answer else []})
        
        # Fallback to alternative field names
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
                        answers = [answer_text] if answer_text else []
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
    documents: Dict[str, Dict[str, Any]] = {}
    questions_by_doc: Dict[str, List[Dict[str, Any]]] = {}
    
    for example in dataset_split:
        story_id = example.get("document_id") or example.get("id") or example.get("example_id") or f"story-{hash(str(example))}"
        
        if story_id not in documents:
            doc_data = {k: v for k, v in example.items() 
                       if k not in ("question", "answers", "answer1", "answer2")}
            documents[story_id] = doc_data
        
        question_text = example.get("question") or example.get("question_text") or ""
        if question_text:
            answers = example.get("answers") or []
            if not answers:
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
    
    for story_id, doc_data in documents.items():
        questions = questions_by_doc.get(story_id, [])
        if not questions:
            logging.debug("No questions found for story %s", story_id)
            continue
        yield story_id, doc_data, questions


def iter_quality_article_questions(dataset_split: Sequence[Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Iterate over QuALITY dataset articles and questions."""
    for example in dataset_split:
        article_id = example.get("article_id") or example.get("id") or f"article-{hash(str(example))}"
        questions: List[Dict[str, Any]] = []
        
        questions_list = example.get("questions", [])
        for q_item in questions_list:
            if isinstance(q_item, dict):
                question_text = q_item.get("question") or q_item.get("question_text") or ""
                if not question_text:
                    continue
                
                options = q_item.get("options", [])
                gold_label = q_item.get("gold_label")
                answers = []
                
                if gold_label is not None and isinstance(options, list) and len(options) > 0:
                    try:
                        gold_idx = int(gold_label) - 1
                        if 0 <= gold_idx < len(options):
                            answers.append(options[gold_idx])
                        else:
                            logging.warning("Invalid gold_label %d for question in article %s", gold_label, article_id)
                            answers = options[:1] if options else []
                    except (ValueError, TypeError):
                        logging.warning("Invalid gold_label type for question in article %s: %s", article_id, gold_label)
                        answers = options[:1] if options else []
                elif options:
                    answers = [options[0]] if options else []
                
                questions.append({"question": question_text, "answers": answers})
        
        if not questions:
            logging.debug("No questions found for article %s", article_id)
            continue
        
        yield article_id, example, questions


def _iter_single_qa_questions(dataset_split: Sequence[Dict[str, Any]], id_fields: List[str], question_field: str = "question") -> Iterable[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Generic iterator for datasets where each entry is a single QA pair."""
    for idx, example in enumerate(dataset_split):
        qa_id = None
        for field in id_fields:
            qa_id = example.get(field)
            if qa_id:
                break
        if not qa_id:
            qa_id = f"entry-{idx}"
        
        question_text = example.get(question_field) or ""
        if not question_text:
            logging.debug("No question found for entry %s", qa_id)
            continue
        
        # Extract answers
        answer = example.get("answer") or ""
        answers_list = example.get("answers") or []
        answers = []
        if answer:
            answers.append(answer)
        if isinstance(answers_list, list):
            answers.extend([str(a) for a in answers_list if a])
        elif isinstance(answers_list, str) and answers_list:
            if answers_list not in answers:
                answers.append(answers_list)
        
        # Remove duplicates
        seen = set()
        unique_answers = [a for a in answers if a and a not in seen and not seen.add(a)]
        
        yield qa_id, example, [{"question": question_text, "answers": unique_answers}]


def iter_hotpot_article_questions(dataset_split: Sequence[Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Iterate over HotpotQA dataset questions."""
    yield from _iter_single_qa_questions(dataset_split, ["_id", "id"], "question")


def iter_musique_article_questions(dataset_split: Sequence[Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Iterate over MuSiQue dataset questions."""
    yield from _iter_single_qa_questions(dataset_split, ["question_id", "id", "_id"], "question")


def iter_xsum_article_questions(dataset_split: Sequence[Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Iterate over XSum dataset articles for summarization."""
    for idx, example in enumerate(dataset_split):
        article_id = example.get("id") or example.get("_id") or f"xsum-{idx}"
        document = example.get("document") or example.get("article") or ""
        summary = example.get("summary") or ""
        
        if not document:
            logging.debug("No document found for XSum entry %s", article_id)
            continue
        
        # For summarization, we use a generic question
        # The reference answer is the summary
        question_text = "Summarize this article."
        answers = [summary] if summary else []
        
        yield article_id, example, [{"question": question_text, "answers": answers}]


def iter_wikiasp_article_questions(dataset_split: Sequence[Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Iterate over WikiAsp dataset articles and aspect-based questions.
    
    WikiAsp format:
    - "exid": example ID
    - "inputs": list of text chunks (article content)
    - "targets": list of [aspect_name, summary_text] pairs
    """
    for idx, example in enumerate(dataset_split):
        # Use exid if available, otherwise fall back to other ID fields
        article_id = example.get("exid") or example.get("id") or example.get("article_id") or example.get("_id") or f"wikiasp-{idx}"
        questions: List[Dict[str, Any]] = []
        
        # WikiAsp format: targets is a list of [aspect_name, summary_text] pairs
        targets = example.get("targets") or []
        
        if isinstance(targets, list):
            for target_item in targets:
                if isinstance(target_item, list) and len(target_item) >= 2:
                    # Format: [aspect_name, summary_text]
                    aspect_name = target_item[0]
                    aspect_summary = target_item[1]
                    if aspect_name and aspect_summary:
                        question_text = f"Summarize the {aspect_name} aspect of this article."
                        # Ensure answers is a list
                        answers = [aspect_summary] if isinstance(aspect_summary, str) else (aspect_summary if isinstance(aspect_summary, list) else [str(aspect_summary)])
                        questions.append({"question": question_text, "answers": answers})
                elif isinstance(target_item, dict):
                    # Alternative dict format
                    aspect_name = target_item.get("aspect") or target_item.get("name") or "aspect"
                    aspect_summary = target_item.get("summary") or target_item.get("text") or ""
                    if aspect_summary:
                        question_text = f"Summarize the {aspect_name} aspect of this article."
                        answers = [aspect_summary] if isinstance(aspect_summary, str) else (aspect_summary if isinstance(aspect_summary, list) else [str(aspect_summary)])
                        questions.append({"question": question_text, "answers": answers})
        
        # Fallback: check for old format (aspects/summaries dicts)
        if not questions:
            aspects = example.get("aspects") or {}
            summaries = example.get("summaries") or {}
            
            if isinstance(aspects, dict) and aspects:
                for aspect_name, aspect_summary in aspects.items():
                    if aspect_summary:
                        question_text = f"Summarize the {aspect_name} aspect of this article."
                        answers = [aspect_summary] if isinstance(aspect_summary, str) else (aspect_summary if isinstance(aspect_summary, list) else [str(aspect_summary)])
                        questions.append({"question": question_text, "answers": answers})
            elif isinstance(summaries, dict) and summaries:
                for aspect_name, aspect_summary in summaries.items():
                    if aspect_summary:
                        question_text = f"Summarize the {aspect_name} aspect of this article."
                        answers = [aspect_summary] if isinstance(aspect_summary, str) else (aspect_summary if isinstance(aspect_summary, list) else [str(aspect_summary)])
                        questions.append({"question": question_text, "answers": answers})
        
        if not questions:
            logging.debug("No targets/aspects/summaries found for WikiAsp entry %s", article_id)
            continue
        
        yield article_id, example, questions


def iter_longbench_article_questions(dataset_split: Sequence[Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Iterate over LongBench dataset questions.
    
    LongBench format:
    - input: The question/query
    - context: The long context document
    - answers: List of true answers
    - dataset: Name of the sub-dataset
    - _id: Unique identifier
    """
    for idx, example in enumerate(dataset_split):
        article_id = example.get("_id") or example.get("id") or f"longbench-{idx}"
        input_text = example.get("input") or ""
        context = example.get("context") or ""
        answers = example.get("answers") or []
        
        if not input_text:
            logging.debug("No input/question found for LongBench entry %s", article_id)
            continue
        
        # Ensure answers is a list
        if isinstance(answers, str):
            answers = [answers]
        elif not isinstance(answers, list):
            answers = [str(answers)] if answers else []
        
        # LongBench already has questions as "input", so we use it directly
        questions = [{"question": input_text, "answers": answers}]
        
        yield article_id, example, questions


def iter_article_questions(dataset_split: Sequence[Dict[str, Any]], dataset_name: str) -> Iterable[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    """Dispatch to appropriate iterator based on dataset name."""
    dataset_lower = dataset_name.lower()
    iterators = {
        "qasper": iter_qasper_article_questions,
        "qmsum": iter_qmsum_article_questions,
        "narrativeqa": iter_narrativeqa_article_questions,
        "quality": iter_quality_article_questions,
        "hotpot": iter_hotpot_article_questions,
        "musique": iter_musique_article_questions,
        "xsum": iter_xsum_article_questions,
        "wikiasp": iter_wikiasp_article_questions,
        "longbench": iter_longbench_article_questions,
    }
    
    iterator_func = iterators.get(dataset_lower, iter_qasper_article_questions)
    yield from iterator_func(dataset_split)

