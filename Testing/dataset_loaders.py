"""Dataset loading functions for various datasets."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from collections.abc import Callable
from typing import Any, Dict, List

try:
    from datasets import load_dataset  # type: ignore
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'datasets'. Install it with `pip install datasets`."
    ) from exc


def _load_jsonl_file(filepath: Path) -> List[Dict[str, Any]]:
    """Helper to load JSONL file."""
    records = []
    try:
        with filepath.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logging.warning("Failed to parse line %d in %s: %s", line_num, filepath, exc)
    except Exception as exc:
        logging.warning("Failed to load %s: %s", filepath, exc)
        raise
    return records


def _load_json_file(filepath: Path) -> List[Dict[str, Any]]:
    """Helper to load JSON file."""
    try:
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                logging.warning("Unexpected data type in %s", filepath)
                return []
    except Exception as exc:
        logging.warning("Failed to load %s: %s", filepath, exc)
        raise


def load_narrativeqa_from_local(split: str, dataset_dir: Path) -> List[Dict[str, Any]]:
    """Load NarrativeQA dataset from local CSV files."""
    split_map = {"validation": "train", "train": "train", "test": "test"}
    split_name = split_map.get(split.lower(), split.lower())
    
    documents_csv = dataset_dir / "documents.csv"
    qaps_csv = dataset_dir / "qaps.csv"
    summaries_csv = dataset_dir / "third_party" / "wikipedia" / "summaries.csv"
    
    if not documents_csv.exists() or not qaps_csv.exists():
        raise FileNotFoundError(f"Required CSV files not found in {dataset_dir}")
    
    # Load documents
    documents = {}
    try:
        with documents_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_id = row.get("document_id")
                doc_set = row.get("set", "").lower()
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
                    if split.lower() == "validation" and summary_set == "train":
                        summaries[doc_id] = row.get("summary", "")
                    elif summary_set == split_name:
                        summaries[doc_id] = row.get("summary", "")
        except Exception as exc:
            logging.warning("Failed to load summaries.csv: %s", exc)
    
    # Combine data
    records: List[Dict[str, Any]] = []
    for doc_id, doc_data in documents.items():
        if doc_id in qas_by_doc:
            for qa in qas_by_doc[doc_id]:
                record = doc_data.copy()
                record["question"] = qa.get("question", "")
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
        raise FileNotFoundError(f"No NarrativeQA records found for split '{split}'")
    
    logging.info("Loaded %d NarrativeQA records from local CSV files", len(records))
    return records


def load_qmsum_from_local(split: str, dataset_dir: Path) -> List[Dict[str, Any]]:
    """Load QMSum dataset from local directory."""
    split_map = {"validation": "val", "train": "train", "test": "test"}
    split_dir = split_map.get(split.lower(), split.lower())
    
    domains = ["ALL", "Academic", "Product", "Committee"]
    records: List[Dict[str, Any]] = []
    
    for domain in domains:
        domain_path = dataset_dir / domain / split_dir
        if not domain_path.exists():
            continue
        
        json_files = list(domain_path.glob("*.json"))
        logging.info("Found %d JSON files in %s/%s", len(json_files), domain, split_dir)
        
        for json_file in json_files:
            try:
                record = _load_json_file(json_file)[0] if _load_json_file(json_file) else {}
                record["meeting_id"] = json_file.stem
                record["domain"] = domain
                record["split"] = split_dir
                records.append(record)
            except Exception as exc:
                logging.warning("Failed to load %s: %s", json_file, exc)
    
    if not records:
        raise FileNotFoundError(f"No QMSum records found in {dataset_dir}")
    
    logging.info("Loaded %d QMSum records from local directory", len(records))
    return records


def load_quality_from_local(split: str, dataset_dir: Path) -> List[Dict[str, Any]]:
    """Load QuALITY dataset from local JSONL files."""
    split_map = {"validation": "dev", "train": "train", "test": "test", "dev": "dev"}
    split_name = split_map.get(split.lower(), split.lower())
    
    versions = ["v1.0.1", "v1.0", "v0.9"]
    records: List[Dict[str, Any]] = []
    
    for version in versions:
        version_dir = dataset_dir / version
        if not version_dir.exists():
            continue
        
        filename_patterns = [
            f"QuALITY.{version}.{split_name}",
            f"QuALITY.{version}.htmlstripped.{split_name}",
        ]
        
        for filename_pattern in filename_patterns:
            jsonl_file = version_dir / filename_pattern
            if jsonl_file.exists():
                logging.info("Found QuALITY file: %s", jsonl_file)
                records = _load_jsonl_file(jsonl_file)
                if records:
                    break
        
        if records:
            break
    
    if not records:
        raise FileNotFoundError(f"No QuALITY records found in {dataset_dir}")
    
    logging.info("Loaded %d QuALITY records from local directory", len(records))
    return records


def load_hotpot_from_local(split: str, dataset_dir: Path) -> List[Dict[str, Any]]:
    """Load HotpotQA dataset from local JSON files."""
    split_map = {"validation": "dev", "train": "train", "test": "test", "dev": "dev"}
    split_name = split_map.get(split.lower(), split.lower())
    
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
            records = _load_json_file(json_file)
            if records:
                break
    
    if not records:
        raise FileNotFoundError(f"No HotpotQA records found in {dataset_dir}")
    
    logging.info("Loaded %d HotpotQA records from local directory", len(records))
    return records


def load_musique_from_local(split: str, dataset_dir: Path) -> List[Dict[str, Any]]:
    """Load MuSiQue dataset from local JSONL files."""
    split_map = {"validation": "dev", "train": "train", "test": "test", "dev": "dev"}
    split_name = split_map.get(split.lower(), split.lower())
    
    data_dir = dataset_dir / "data" if (dataset_dir / "data").exists() else dataset_dir
    records: List[Dict[str, Any]] = []
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
                if jsonl_file.suffix == ".jsonl":
                    records = _load_jsonl_file(jsonl_file)
                else:
                    records = _load_json_file(jsonl_file)
                if records:
                    break
        
        if records:
            break
    
    if not records:
        raise FileNotFoundError(f"No MuSiQue records found in {dataset_dir} or {data_dir}")
    
    logging.info("Loaded %d MuSiQue records from local directory", len(records))
    return records


def load_xsum_from_local(split: str, dataset_dir: Path) -> List[Dict[str, Any]]:
    """Load XSum dataset from local JSON files or JSONL files."""
    split_map = {"validation": "validation", "train": "train", "test": "test", "dev": "validation"}
    split_name = split_map.get(split.lower(), split.lower())
    
    data_dir = dataset_dir / "data" if (dataset_dir / "data").exists() else dataset_dir
    records: List[Dict[str, Any]] = []
    
    filename_patterns = [
        f"xsum.{split_name}.jsonl",
        f"xsum_{split_name}.jsonl",
        f"{split_name}.jsonl",
        f"xsum.{split_name}.json",
        f"xsum_{split_name}.json",
        f"{split_name}.json",
    ]
    
    for filename_pattern in filename_patterns:
        file_path = data_dir / filename_pattern
        if file_path.exists():
            logging.info("Found XSum file: %s", file_path)
            if file_path.suffix == ".jsonl":
                records = _load_jsonl_file(file_path)
            else:
                records = _load_json_file(file_path)
            if records:
                break
    
    if not records:
        raise FileNotFoundError(f"No XSum records found in {dataset_dir} or {data_dir}. Tried splits: {split_name}")
    
    logging.info("Loaded %d XSum records from local directory", len(records))
    return records


def load_wikiasp_from_local(split: str, dataset_dir: Path) -> List[Dict[str, Any]]:
    """Load WikiAsp dataset from local JSON files or JSONL files.
    
    WikiAsp has multiple topic folders (Album, Animal, Artist, Building, etc.)
    Each folder contains train.jsonl, test.jsonl, and valid.jsonl files.
    """
    # WikiAsp uses "valid" for validation, not "dev"
    split_map = {"validation": "valid", "train": "train", "test": "test", "dev": "valid", "valid": "valid"}
    split_name = split_map.get(split.lower(), split.lower())
    
    # WikiAsp has domain-specific subdirectories in the data folder
    data_dir = dataset_dir / "data" if (dataset_dir / "data").exists() else dataset_dir
    records: List[Dict[str, Any]] = []
    
    # Try common filename patterns in root data directory first (fallback)
    filename_patterns = [
        f"wikiasp.{split_name}.jsonl",
        f"wikiasp_{split_name}.jsonl",
        f"{split_name}.jsonl",
        f"wikiasp.{split_name}.json",
        f"wikiasp_{split_name}.json",
        f"{split_name}.json",
    ]
    
    for filename_pattern in filename_patterns:
        file_path = data_dir / filename_pattern
        if file_path.exists():
            logging.info("Found WikiAsp file: %s", file_path)
            if file_path.suffix == ".jsonl":
                records = _load_jsonl_file(file_path)
            else:
                records = _load_json_file(file_path)
            if records:
                break
    
    # Load from all topic subdirectories (WikiAsp has multiple topics like Album, Animal, Artist, etc.)
    if not records:
        topic_dirs = [d for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if topic_dirs:
            logging.info("Found %d topic directories in WikiAsp data folder", len(topic_dirs))
            for topic_dir in sorted(topic_dirs):
                # Look for split file in this topic directory
                topic_file = topic_dir / f"{split_name}.jsonl"
                if topic_file.exists():
                    logging.info("Loading WikiAsp records from topic %s: %s", topic_dir.name, topic_file)
                    try:
                        topic_records = _load_jsonl_file(topic_file)
                        if topic_records:
                            # Add topic information to each record
                            for record in topic_records:
                                record['topic'] = topic_dir.name
                            records.extend(topic_records)
                            logging.info("Loaded %d records from topic %s (total: %d)", 
                                       len(topic_records), topic_dir.name, len(records))
                    except Exception as exc:
                        logging.warning("Failed to load WikiAsp file from topic %s: %s", topic_dir.name, exc)
    
    if not records:
        raise FileNotFoundError(f"No WikiAsp records found in {dataset_dir} or {data_dir}. Tried splits: {split_name}")
    
    logging.info("Loaded %d total WikiAsp records from %d topic(s)", len(records), 
                len(set(r.get('topic', 'unknown') for r in records)))
    return records


def load_longbench_from_local(split: str, dataset_dir: Path) -> List[Dict[str, Any]]:
    """Load LongBench dataset from local JSONL files.
    
    LongBench has multiple sub-datasets. This function loads all available JSONL files
    from the data directory or combines them into a single list.
    """
    split_map = {"validation": "test", "train": "train", "test": "test", "dev": "test"}
    split_name = split_map.get(split.lower(), "test")
    
    # LongBench data is typically in a data directory
    data_dir = dataset_dir / "data" if (dataset_dir / "data").exists() else dataset_dir
    records: List[Dict[str, Any]] = []
    
    # LongBench sub-datasets include:
    # narrativeqa, qasper, multifieldqa_en, multifieldqa_zh, hotpotqa, 2wikimqa, musique,
    # dureader, gov_report, qmsum, multi_news, vcsum, trec, triviaqa, samsum, lsht,
    # passage_count, passage_retrieval_en, passage_retrieval_zh, lcc, repobench-p
    # And their LongBench-E variants (_e suffix)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"LongBench data directory not found: {data_dir}")
    
    # Collect all JSONL files in the data directory
    jsonl_files = list(data_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in LongBench data directory: {data_dir}")
    
    # Load all sub-datasets (or you could filter by specific sub-dataset if needed)
    for jsonl_file in sorted(jsonl_files):
        try:
            file_records = _load_jsonl_file(jsonl_file)
            if file_records:
                logging.info("Loaded %d records from LongBench sub-dataset: %s", len(file_records), jsonl_file.name)
                records.extend(file_records)
        except Exception as exc:
            logging.warning("Failed to load LongBench file %s: %s", jsonl_file, exc)
    
    if not records:
        raise FileNotFoundError(f"No LongBench records found in {data_dir}")
    
    logging.info("Loaded %d total LongBench records from local directory", len(records))
    return records


def load_dataset_with_fallback(
    dataset_name: str,
    split: str,
    local_dir: Path,
    hf_dataset_name: str | None = None,
    load_local_func: Callable[[str, Path], List[Dict[str, Any]]] | None = None
) -> List[Dict[str, Any]]:
    """Load dataset from local directory with HuggingFace fallback."""
    dataset_name_lower = dataset_name.lower()
    
    # Try local first
    if local_dir.exists() and load_local_func:
        try:
            logging.info("Loading %s dataset split '%s' from local directory...", dataset_name, split)
            dataset_split = load_local_func(split, local_dir)
            logging.info("Loaded %d records from local %s directory", len(dataset_split), dataset_name)
            return dataset_split
        except Exception as exc:
            logging.error("Failed to load %s from local directory: %s", dataset_name, exc)
            if hf_dataset_name:
                logging.info("Falling back to HuggingFace dataset...")
            else:
                raise FileNotFoundError(f"Could not load {dataset_name} from local directory. Error: {exc}")
    
    # Try HuggingFace
    if hf_dataset_name:
        try:
            dataset_split = load_dataset(hf_dataset_name, split=split)
            logging.info("Loaded %d records from HuggingFace %s", len(dataset_split), dataset_name)
            return list(dataset_split)
        except Exception as exc:
            logging.error("Failed to load %s from HuggingFace: %s", dataset_name, exc)
            raise FileNotFoundError(
                f"Could not load {dataset_name} dataset. Local directory not found at {local_dir} "
                f"and HuggingFace load failed: {exc}"
            )
    
    raise FileNotFoundError(f"{dataset_name} directory not found at {local_dir}")

