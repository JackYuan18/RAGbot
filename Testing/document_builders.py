"""Document building functions for extracting text from various dataset formats."""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Tuple

from RAGSystem import RAGSystem  # type: ignore


class HTMLStripper(HTMLParser):
    """Simple HTML parser to extract text content."""
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip = False
    
    def handle_starttag(self, tag, attrs):
        if tag.lower() in ('script', 'style'):
            self.skip = True
    
    def handle_endtag(self, tag):
        if tag.lower() in ('script', 'style'):
            self.skip = False
        elif tag.lower() in ('p', 'div', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            self.text.append('\n')
    
    def handle_data(self, data):
        if not self.skip:
            self.text.append(data)


def _create_text_adder(
    rag_system: RAGSystem,
    article_id: str,
    chunk_size: int,
    title: str | None,
    records: Dict[str, Any],
    documents: List[str],
    doc_metadata: List[Dict[str, Any]]
):
    """Create a closure for adding text blocks with chunking."""
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
            doc_metadata.append({
                "article_id": article_id,
                "section_title": section_title,
                "paragraph_index": paragraph_index,
                "chunk_offset": local_idx,
                "text_preview": chunk[:200] + ("..." if len(chunk) > 200 else ""),
                "title": title or f"Document {article_id}",
            })
    return add_text_block


def build_qmsum_documents(
    rag_system: RAGSystem,
    meeting_id: str,
    record: Dict[str, Any],
    chunk_size: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Build documents from QMSum meeting transcript."""
    documents: List[str] = []
    doc_metadata: List[Dict[str, Any]] = []
    title = record.get("meeting") or record.get("title") or f"Meeting {meeting_id}"
    add_text_block = _create_text_adder(rag_system, meeting_id, chunk_size, title, record, documents, doc_metadata)
    
    transcripts = record.get("meeting_transcripts") or record.get("transcript") or record.get("text")
    
    if isinstance(transcripts, list):
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
        paragraphs = transcripts.split("\n\n") or [transcripts]
        for idx, para in enumerate(paragraphs):
            if para.strip():
                add_text_block(para.strip(), None, idx)
    
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
    title = record.get("document_title") or record.get("title") or f"Story {story_id}"
    add_text_block = _create_text_adder(rag_system, story_id, chunk_size, title, record, documents, doc_metadata)
    
    story_text = record.get("document_text") or record.get("story") or record.get("text") or record.get("article")
    
    if isinstance(story_text, str) and story_text.strip():
        paragraphs = story_text.split("\n\n")
        if len(paragraphs) == 1:
            paragraphs = story_text.split("\n")
        for idx, para in enumerate(paragraphs):
            if para.strip():
                add_text_block(para.strip(), None, idx)
    elif isinstance(story_text, list):
        for idx, para in enumerate(story_text):
            if isinstance(para, str) and para.strip():
                add_text_block(para.strip(), None, idx)
            elif isinstance(para, dict):
                para_text = para.get("text") or para.get("content") or para.get("paragraph") or ""
                if para_text.strip():
                    section = para.get("section") or para.get("heading")
                    add_text_block(para_text.strip(), section, idx)
    
    summary = record.get("summary") or record.get("summary_text") or record.get("document_summary")
    if isinstance(summary, str) and summary.strip():
        add_text_block(summary.strip(), "Summary", 0)
    elif isinstance(summary, dict):
        summary_text = summary.get("text") or summary.get("summary") or ""
        if summary_text.strip():
            add_text_block(summary_text.strip(), "Summary", 0)
    
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
    documents: List[str] = []
    doc_metadata: List[Dict[str, Any]] = []
    title = record.get("title") or f"Article {article_id}"
    add_text_block = _create_text_adder(rag_system, article_id, chunk_size, title, record, documents, doc_metadata)
    
    article_html = record.get("article") or record.get("html") or ""
    
    if article_html:
        stripper = HTMLStripper()
        try:
            stripper.feed(article_html)
            article_text = ''.join(stripper.text)
            article_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', article_text)
            article_text = re.sub(r' +', ' ', article_text)
            
            paragraphs = [p.strip() for p in article_text.split('\n\n') if p.strip()]
            current_section = None
            for idx, para in enumerate(paragraphs):
                if para.strip():
                    if len(para) < 100 and not para.rstrip().endswith(('.', '!', '?', ':')) and not para.strip().startswith('<'):
                        current_section = para.strip()
                        continue
                    add_text_block(para.strip(), current_section, idx)
        except Exception as exc:
            logging.warning("Failed to parse HTML for article %s: %s. Using raw text.", article_id, exc)
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
    title = f"HotpotQA {article_id}"
    add_text_block = _create_text_adder(rag_system, article_id, chunk_size, title, record, documents, doc_metadata)
    
    context = record.get("context") or []
    if isinstance(context, list):
        for para_idx, paragraph in enumerate(context):
            if isinstance(paragraph, list) and len(paragraph) >= 2:
                title_para = paragraph[0] if paragraph[0] else None
                sentences = paragraph[1] if isinstance(paragraph[1], list) else []
                if sentences:
                    para_text = " ".join(str(s) for s in sentences if s)
                    if para_text.strip():
                        add_text_block(para_text.strip(), title_para, para_idx)
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
    title = f"MuSiQue {article_id}"
    add_text_block = _create_text_adder(rag_system, article_id, chunk_size, title, record, documents, doc_metadata)
    
    paragraphs = record.get("paragraphs") or []
    if isinstance(paragraphs, list):
        for para_idx, paragraph in enumerate(paragraphs):
            if isinstance(paragraph, dict):
                title_para = paragraph.get("title") or paragraph.get("paragraph_title")
                para_text = paragraph.get("text") or paragraph.get("paragraph_text") or ""
                if para_text:
                    add_text_block(str(para_text).strip(), title_para, para_idx)
            elif isinstance(paragraph, str):
                add_text_block(paragraph.strip(), None, para_idx)
    
    if not documents:
        logging.warning("No textual content extracted for MuSiQue entry %s", article_id)
    
    return documents, doc_metadata


def build_xsum_documents(
    rag_system: RAGSystem,
    article_id: str,
    record: Dict[str, Any],
    chunk_size: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Build documents from XSum article."""
    documents: List[str] = []
    doc_metadata: List[Dict[str, Any]] = []
    title = record.get("title") or f"XSum {article_id}"
    add_text_block = _create_text_adder(rag_system, article_id, chunk_size, title, record, documents, doc_metadata)
    
    document_text = record.get("document") or record.get("article") or ""
    
    if document_text:
        # Split document into paragraphs
        paragraphs = document_text.split("\n\n") or [document_text]
        for para_idx, para in enumerate(paragraphs):
            if para.strip():
                add_text_block(para.strip(), None, para_idx)
    
    if not documents:
        logging.warning("No textual content extracted for XSum entry %s", article_id)
    
    return documents, doc_metadata


def build_wikiasp_documents(
    rag_system: RAGSystem,
    article_id: str,
    record: Dict[str, Any],
    chunk_size: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Build documents from WikiAsp article.
    
    WikiAsp format:
    - "exid": example ID
    - "inputs": list of text chunks (article content)
    - "targets": list of [aspect_name, summary_text] pairs
    - "topic": topic name (added by loader, e.g., "Album", "Animal")
    """
    documents: List[str] = []
    doc_metadata: List[Dict[str, Any]] = []
    
    # Get topic from record (added by loader)
    topic = record.get("topic", "WikiAsp")
    title = f"{topic} {article_id}"
    add_text_block = _create_text_adder(rag_system, article_id, chunk_size, title, record, documents, doc_metadata)
    
    # WikiAsp uses "inputs" field which is a list of text chunks
    inputs = record.get("inputs") or []
    
    if isinstance(inputs, list):
        # Process each input chunk
        for input_idx, input_chunk in enumerate(inputs):
            if input_chunk and str(input_chunk).strip():
                # Remove < EOT > marker if present
                text = str(input_chunk).strip()
                text = text.replace("< EOT >", "").strip()
                if text:
                    add_text_block(text, None, input_idx)
    
    # Fallback: try old format fields if inputs not found
    if not documents:
        main_text = record.get("text") or record.get("article_text") or record.get("content") or ""
        if main_text:
            paragraphs = main_text.split("\n\n") if isinstance(main_text, str) else [main_text]
            for para_idx, para in enumerate(paragraphs):
                if para and str(para).strip():
                    add_text_block(str(para).strip(), None, para_idx)
        
        # Also check for reference documents
        references = record.get("references") or record.get("cited_references") or record.get("documents") or []
        if isinstance(references, list):
            for ref_idx, ref in enumerate(references):
                if isinstance(ref, dict):
                    ref_text = ref.get("text") or ref.get("content") or ref.get("document") or ""
                    ref_title = ref.get("title") or ref.get("name") or f"Reference {ref_idx + 1}"
                    if ref_text:
                        paragraphs = ref_text.split("\n\n") if isinstance(ref_text, str) else [ref_text]
                        for para_idx, para in enumerate(paragraphs):
                            if para and str(para).strip():
                                add_text_block(str(para).strip(), ref_title, para_idx)
                elif isinstance(ref, str) and ref.strip():
                    add_text_block(ref.strip(), f"Reference {ref_idx + 1}", ref_idx)
    
    if not documents:
        logging.warning("No textual content extracted for WikiAsp entry %s", article_id)
    
    return documents, doc_metadata


def build_longbench_documents(
    rag_system: RAGSystem,
    article_id: str,
    record: Dict[str, Any],
    chunk_size: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Build documents from LongBench entry.
    
    LongBench format:
    - context: The long context document(s) required for the task
    - dataset: Name of the sub-dataset (e.g., "narrativeqa", "qasper", etc.)
    """
    documents: List[str] = []
    doc_metadata: List[Dict[str, Any]] = []
    sub_dataset = record.get("dataset") or "LongBench"
    title = f"LongBench {sub_dataset} {article_id}"
    add_text_block = _create_text_adder(rag_system, article_id, chunk_size, title, record, documents, doc_metadata)
    
    # LongBench context contains the long document(s) needed for the task
    context = record.get("context") or ""
    
    if context:
        # Split context into paragraphs
        paragraphs = context.split("\n\n") if isinstance(context, str) else [context]
        for para_idx, para in enumerate(paragraphs):
            if para and str(para).strip():
                add_text_block(str(para).strip(), None, para_idx)
    
    if not documents:
        logging.warning("No textual content extracted for LongBench entry %s", article_id)
    
    return documents, doc_metadata


def build_qasper_documents(
    rag_system: RAGSystem,
    article_id: str,
    record: Dict[str, Any],
    chunk_size: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Build documents from QASPER article."""
    documents: List[str] = []
    doc_metadata: List[Dict[str, Any]] = []
    title = record.get("title")
    add_text_block = _create_text_adder(rag_system, article_id, chunk_size, title, record, documents, doc_metadata)
    
    abstract = record.get("abstract")
    if isinstance(abstract, str):
        add_text_block(abstract, "Abstract", 0)
    elif isinstance(abstract, list):
        for idx, paragraph in enumerate(abstract):
            add_text_block(paragraph, "Abstract", idx)
    
    metadata = record.get("metadata")
    full_text = metadata.get("full_text") or metadata.get("fullText") if isinstance(metadata, dict) else None
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
            section_paragraphs: list = []
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


def build_article_documents(
    rag_system: RAGSystem,
    article_id: str,
    record: Dict[str, Any],
    chunk_size: int,
    dataset_name: str = "qasper",
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Dispatch to appropriate document builder based on dataset name."""
    dataset_lower = dataset_name.lower()
    builders = {
        "qasper": build_qasper_documents,
        "qmsum": build_qmsum_documents,
        "narrativeqa": build_narrativeqa_documents,
        "quality": build_quality_documents,
        "hotpot": build_hotpot_documents,
        "musique": build_musique_documents,
        "xsum": build_xsum_documents,
        "wikiasp": build_wikiasp_documents,
        "longbench": build_longbench_documents,
    }
    
    builder_func = builders.get(dataset_lower, build_qasper_documents)
    return builder_func(rag_system, article_id, record, chunk_size)

