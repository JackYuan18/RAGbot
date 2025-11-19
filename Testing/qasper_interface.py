#!/usr/bin/env python3
"""
Interactive viewer for QASPER RAG evaluation results.

Launch this script to serve a lightweight web interface where you can trigger
`test_qasper_rag.py`, inspect generated answers, and compare them with reference
answers from the dataset.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template_string, request  # type: ignore

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_PATH = CURRENT_DIR / "qasper_results.json"
DEFAULT_TEST_SCRIPT = CURRENT_DIR / "test_qasper_rag.py"

app = Flask(__name__)

# Shared state for UI polling
RUN_LOCK = threading.Lock()
RUN_STATE: dict[str, Any] = {
    "status": "idle",
    "message": "Idle. Click 'Run Tests' to generate fresh results.",
    "last_result": None,
}

RESULTS_PATH: Path = DEFAULT_RESULTS_PATH
TEST_OPTIONS: argparse.Namespace


def run_test_script(options: argparse.Namespace) -> subprocess.CompletedProcess[str]:
    """Run test script for a single dataset."""
    if not DEFAULT_TEST_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot locate test script at {DEFAULT_TEST_SCRIPT}")

    dataset = getattr(options, "dataset", "qasper")
    # Ensure RESULTS_PATH is absolute
    results_path_abs = RESULTS_PATH.resolve()
    results_path_abs.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable,
        str(DEFAULT_TEST_SCRIPT),
        "--dataset",
        dataset,
        "--split",
        options.split,
        "--articles",
        str(options.articles),
        "--questions-per-article",
        str(options.questions_per_article),
        "--retrieval-k",
        str(options.retrieval_k),
        "--chunk-size",
        str(options.chunk_size),
        "--generator-model",
        options.generator_model,
        "--log-level",
        options.log_level,
        "--save-json",
        str(results_path_abs),
    ]

    if getattr(options, "chatgpt5_api_key", None):
        cmd.extend(["--chatgpt5-api-key", options.chatgpt5_api_key])

    if getattr(options, "show_context", False):
        cmd.append("--show-context")
    else:
        cmd.append("--hide-context")

    display_cmd = list(cmd)
    if "--chatgpt5-api-key" in display_cmd:
        try:
            idx = display_cmd.index("--chatgpt5-api-key")
            if idx + 1 < len(display_cmd):
                display_cmd[idx + 1] = "****"
        except ValueError:
            pass

    app.logger.info("Running test script: %s", " ".join(display_cmd))
    app.logger.info("Results will be saved to: %s (absolute: %s)", RESULTS_PATH, results_path_abs)
    app.logger.info("Working directory: %s", CURRENT_DIR)
    app.logger.info("Python executable: %s", sys.executable)
    app.logger.info("Test script path: %s (exists: %s)", DEFAULT_TEST_SCRIPT, DEFAULT_TEST_SCRIPT.exists())
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, cwd=str(CURRENT_DIR), timeout=3600)
    except subprocess.TimeoutExpired:
        app.logger.error("Test script timed out after 1 hour")
        return subprocess.CompletedProcess(cmd, 1, "", "Test script timed out")
    except Exception as exc:
        app.logger.error("Failed to run test script: %s", exc)
        return subprocess.CompletedProcess(cmd, 1, "", str(exc))
    
    if result.returncode != 0:
        app.logger.error("Test script failed with return code %d", result.returncode)
        app.logger.error("STDOUT: %s", result.stdout[-1000:] if result.stdout else "(empty)")
        app.logger.error("STDERR: %s", result.stderr[-1000:] if result.stderr else "(empty)")
    else:
        app.logger.info("Test script completed successfully")
        if results_path_abs.exists():
            app.logger.info("Results file exists at: %s", results_path_abs)
            try:
                with results_path_abs.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    app.logger.info("Results file contains %d entries", len(data) if isinstance(data, list) else 0)
            except Exception as e:
                app.logger.error("Failed to read results file: %s", e)
        else:
            app.logger.warning("Results file not found at: %s", results_path_abs)
    return result


def run_multiple_datasets(options: argparse.Namespace, datasets: List[str]) -> subprocess.CompletedProcess[str]:
    """Run test script for multiple datasets sequentially and combine results."""
    if not DEFAULT_TEST_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot locate test script at {DEFAULT_TEST_SCRIPT}")

    all_results: List[Dict[str, Any]] = []
    combined_output = []
    combined_error = []
    last_returncode = 0

    # Ensure RESULTS_PATH is absolute
    results_path_abs = RESULTS_PATH.resolve()
    results_path_abs.parent.mkdir(parents=True, exist_ok=True)

    # Clear results file at the start
    clear_results_file()

    for idx, dataset in enumerate(datasets):
        app.logger.info("Running tests for dataset %d/%d: %s", idx + 1, len(datasets), dataset)
        RUN_STATE["message"] = f"Running tests for dataset {idx + 1}/{len(datasets)}: {dataset}..."
        
        # Create options for this dataset
        dataset_options = argparse.Namespace(**vars(options))
        dataset_options.dataset = dataset
        
        # Run test for this dataset (this will write to RESULTS_PATH)
        result = run_test_script(dataset_options)
        
        # Collect output
        if result.stdout:
            combined_output.append(f"\n=== Dataset: {dataset} ===\n")
            combined_output.append(result.stdout)
        if result.stderr:
            combined_error.append(f"\n=== Dataset: {dataset} ===\n")
            combined_error.append(result.stderr)
        
        # Update last returncode (keep error if any dataset fails)
        if result.returncode != 0:
            last_returncode = result.returncode
        
        # Load results from this dataset run and add to combined results
        try:
            if results_path_abs.exists():
                with results_path_abs.open("r", encoding="utf-8") as f:
                    dataset_results = json.load(f)
                    if isinstance(dataset_results, list):
                        all_results.extend(dataset_results)
                        app.logger.info("Loaded %d results from dataset %s (total: %d)", len(dataset_results), dataset, len(all_results))
                    else:
                        app.logger.warning("Results file for dataset %s does not contain a list", dataset)
            else:
                app.logger.warning("Results file not found after dataset %s run: %s", dataset, results_path_abs)
        except Exception as exc:
            app.logger.error("Failed to load results for dataset %s: %s", dataset, exc)
    
    # Save combined results
    try:
        results_path_abs.parent.mkdir(parents=True, exist_ok=True)
        with results_path_abs.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        app.logger.info("Combined %d results from %d datasets into %s", len(all_results), len(datasets), results_path_abs)
        if not all_results:
            app.logger.warning("No results collected from any dataset!")
    except Exception as exc:
        app.logger.error("Failed to save combined results to %s: %s", results_path_abs, exc)
        raise
    
    # Return combined result
    return subprocess.CompletedProcess(
        args=[],
        returncode=last_returncode,
        stdout="".join(combined_output),
        stderr="".join(combined_error),
    )


def start_test_run(options: argparse.Namespace, *, async_run: bool = True) -> bool:
    if not RUN_LOCK.acquire(blocking=False):
        app.logger.warning("Cannot start test run: another run is already in progress")
        return False

    RUN_STATE["status"] = "running"
    RUN_STATE["message"] = "Test run in progress..."
    RUN_STATE["last_result"] = None
    app.logger.info("Starting test run with options: dataset=%s, split=%s, articles=%s, questions_per_article=%s",
                   getattr(options, "dataset", "unknown"), getattr(options, "split", "unknown"),
                   getattr(options, "articles", "unknown"), getattr(options, "questions_per_article", "unknown"))

    def runner() -> None:
        try:
            app.logger.info("Test runner thread started")
            # Check if multiple datasets are requested
            datasets = getattr(options, "datasets", None)
            app.logger.info("Datasets attribute: %s (type: %s)", datasets, type(datasets))
            if datasets and isinstance(datasets, list) and len(datasets) > 1:
                app.logger.info("Running multiple datasets: %s", datasets)
                result = run_multiple_datasets(options, datasets)
            else:
                dataset = getattr(options, "dataset", "qasper")
                app.logger.info("Running single dataset: %s", dataset)
                result = run_test_script(options)
            
            app.logger.info("Test script finished with return code: %d", result.returncode)
            if result.returncode == 0:
                dataset_count = len(datasets) if datasets and isinstance(datasets, list) else 1
                RUN_STATE["message"] = f"Last run completed successfully ({dataset_count} dataset(s))."
                RUN_STATE["last_result"] = "success"
                app.logger.info("test_qasper_rag.py completed successfully.")
                # Verify results file was created
                results_path_abs = RESULTS_PATH.resolve()
                if results_path_abs.exists():
                    try:
                        with results_path_abs.open("r", encoding="utf-8") as f:
                            data = json.load(f)
                            count = len(data) if isinstance(data, list) else 0
                            app.logger.info("Results file verified: %d entries", count)
                    except Exception as e:
                        app.logger.error("Failed to verify results file: %s", e)
                else:
                    app.logger.warning("Results file missing after successful run: %s", results_path_abs)
            else:
                error_msg = f"Run failed with code {result.returncode}."
                if result.stderr:
                    error_msg += f" Error: {result.stderr[:200]}"
                RUN_STATE["message"] = error_msg
                RUN_STATE["last_result"] = "error"
                app.logger.error("test_qasper_rag.py failed (%s):\nSTDOUT:\n%s\nSTDERR:\n%s", 
                               result.returncode, result.stdout[-500:] if result.stdout else "(empty)", 
                               result.stderr[-500:] if result.stderr else "(empty)")
        except Exception as exc:  # pragma: no cover - defensive logging
            error_msg = f"Error: {str(exc)}"
            RUN_STATE["message"] = error_msg
            RUN_STATE["last_result"] = "error"
            app.logger.exception("Unexpected error during test run: %s", exc)
        finally:
            RUN_STATE["status"] = "idle"
            RUN_LOCK.release()
            app.logger.info("Test runner thread finished")

    if async_run:
        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        app.logger.info("Test runner thread started (daemon=%s)", thread.daemon)
    else:
        app.logger.info("Running test synchronously")
        runner()

    return True


def load_results() -> List[dict[str, Any]]:
    results_path_abs = RESULTS_PATH.resolve()
    if results_path_abs.exists():
        try:
            with results_path_abs.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    app.logger.debug("Loaded %d results from %s", len(data), results_path_abs)
                    return data
                else:
                    app.logger.warning("Results file does not contain a list: %s", results_path_abs)
        except Exception as exc:  # pragma: no cover - defensive logging
            app.logger.error("Failed to read %s: %s", results_path_abs, exc)
    else:
        app.logger.debug("Results file does not exist: %s", results_path_abs)
    return []


def calculate_dataset_statistics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate statistics (average, min, max) for each dataset.
    
    Returns a dictionary mapping dataset name to statistics dict.
    """
    from collections import defaultdict
    
    # Group results by dataset
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        dataset = result.get("dataset", "unknown")
        by_dataset[dataset].append(result)
    
    statistics: Dict[str, Dict[str, Any]] = {}
    
    for dataset, dataset_results in by_dataset.items():
        metrics = {
            "exact_match": [],
            "f1_score": [],
            "recall_score": [],
            "rouge_l_score": [],
            "bleu_score": [],
        }
        
        # Collect all scores
        for result in dataset_results:
            for metric_name in metrics.keys():
                score = result.get(metric_name)
                if score is not None:
                    try:
                        score_float = float(score)
                        if not (score_float is None or (isinstance(score_float, float) and score_float != score_float)):  # Check for NaN
                            metrics[metric_name].append(score_float)
                    except (TypeError, ValueError):
                        pass
        
        # Calculate total questions in dataset and available (from first result's metadata, all should have same value)
        total_questions_in_dataset = None
        total_questions_available = None
        if dataset_results:
            first_result = dataset_results[0]
            total_questions_in_dataset = first_result.get("total_questions_in_dataset")
            total_questions_available = first_result.get("total_questions_available")
        
        # Calculate statistics for each metric
        stats: Dict[str, Any] = {
            "count": len(dataset_results),
            "total_questions_in_dataset": total_questions_in_dataset,
            "total_questions_available": total_questions_available,
            "metrics": {}
        }
        
        for metric_name, scores in metrics.items():
            if scores:
                stats["metrics"][metric_name] = {
                    "average": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores),
                }
            else:
                stats["metrics"][metric_name] = {
                    "average": None,
                    "min": None,
                    "max": None,
                    "count": 0,
                }
        
        statistics[dataset] = stats
    
    return statistics


def clear_results_file() -> None:
    try:
        RESULTS_PATH.write_text("[]", encoding="utf-8")
    except Exception as exc:
        app.logger.error("Failed to clear results file %s: %s", RESULTS_PATH, exc)


TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Qasper RAG Test Results</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; background: #f8f9fa; color: #212529; }
    h1 { margin-bottom: 0.5rem; }
    .meta { margin-bottom: 1.5rem; color: #495057; }
    .actions { display: flex; gap: 1rem; align-items: center; margin-bottom: 1.5rem; flex-wrap: wrap; }
    button { padding: 0.6rem 1.2rem; border: none; border-radius: 0.3rem; background: #0d6efd; color: #fff; font-size: 1rem; cursor: pointer; }
    button:disabled { background: #6c757d; cursor: not-allowed; }
    #run-status { font-style: italic; color: #495057; }
    table { width: 100%; border-collapse: collapse; background: #fff; box-shadow: 0 0 12px rgba(0,0,0,0.08); }
    th, td { padding: 0.85rem; border-bottom: 1px solid #dee2e6; vertical-align: top; }
    th { background: #e9ecef; text-align: left; }
    tr:nth-child(even) { background: #fefefe; }
    .question { font-weight: 600; }
    .answer { white-space: pre-wrap; }
    ul { margin: 0; padding-left: 1.25rem; }
    .badge { display: inline-block; padding: 0.25rem 0.55rem; margin-bottom: 0.5rem; background: #20c997; color: #fff; border-radius: 0.3rem; font-size: 0.8rem; }
    .context { font-size: 0.9rem; color: #495057; margin-top: 0.6rem; }
    .context-item { margin-bottom: 0.4rem; }
    .config-form { display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem; background: #ffffff; border: 1px solid #dee2e6; border-radius: 0.5rem; padding: 0.75rem 1rem; box-shadow: 0 0 6px rgba(0,0,0,0.05); }
    .config-form label { display: flex; flex-direction: column; align-items: flex-start; gap: 0.35rem; font-size: 0.95rem; color: #495057; }
    .param-input { width: 6rem; padding: 0.35rem 0.5rem; border: 1px solid #ced4da; border-radius: 0.35rem; background: #f8f9fa; color: #6c757d; transition: color 0.2s ease, border-color 0.2s ease, background 0.2s ease; }
    .config-form select.param-input { width: 11rem; }
    .param-input.modified { background: #fff; color: #212529; border-color: #495057; }
    .config-value { color: #6c757d; transition: color 0.2s ease; }
    .config-value.modified { color: #212529; }
    .api-key-input { width: 16rem; padding: 0.35rem 0.5rem; border: 1px solid #ced4da; border-radius: 0.35rem; }
    .api-key-actions { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
    .api-key-actions button { padding: 0.35rem 0.75rem; border-radius: 0.35rem; background: #198754; border: none; color: #fff; cursor: pointer; font-size: 0.9rem; }
    .api-key-actions button:disabled { background: #6c757d; }
    .api-key-status { font-size: 0.9rem; color: #495057; }
    .api-key-status.valid { color: #198754; }
    .api-key-status.invalid { color: #dc3545; }
    .dataset-tab { margin-bottom: 1.5rem; background: #ffffff; border: 1px solid #dee2e6; border-radius: 0.5rem; box-shadow: 0 0 6px rgba(0,0,0,0.05); }
    .tab-header { padding: 1rem 1.25rem; background: #e9ecef; border-bottom: 1px solid #dee2e6; cursor: pointer; user-select: none; border-radius: 0.5rem 0.5rem 0 0; }
    .tab-header:hover { background: #dee2e6; }
    .tab-header-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
    .tab-title { font-weight: 600; font-size: 1.1rem; color: #212529; }
    .tab-toggle { font-size: 1.2rem; color: #495057; transition: transform 0.2s ease; }
    .tab-header.collapsed .tab-toggle { transform: rotate(-90deg); }
    .tab-stats { display: flex; gap: 1.5rem; flex-wrap: wrap; font-size: 0.9rem; }
    .tab-stat-item { display: flex; align-items: center; gap: 0.4rem; }
    .tab-stat-label { color: #495057; font-weight: 500; }
    .tab-stat-value { color: #212529; font-weight: 600; }
    .tab-content { padding: 0; display: none; }
    .tab-content.expanded { display: block; }
    .tab-results-table { width: 100%; border-collapse: collapse; background: #fff; }
    .tab-results-table th, .tab-results-table td { padding: 0.85rem; border-bottom: 1px solid #dee2e6; vertical-align: top; }
    .tab-results-table th { background: #f8f9fa; text-align: left; font-weight: 600; }
    .tab-results-table tbody tr:nth-child(even) { background: #fefefe; }
    .tab-empty-state { padding: 2rem; text-align: center; color: #6c757d; }
    .summary-table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
    .summary-table th, .summary-table td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #dee2e6; }
    .summary-table th { background: #f8f9fa; font-weight: 600; color: #495057; }
    .summary-table td { color: #212529; }
    .summary-table tr:last-child td { border-bottom: none; }
    .metric-value { font-weight: 500; }
    .metric-value.avg { color: #0d6efd; }
    .metric-value.min { color: #dc3545; }
    .metric-value.max { color: #198754; }
    .metric-na { color: #6c757d; font-style: italic; }
  </style>
</head>
<body>
  {% set generator_label = 'ChatGPT 5' if config.generator_model == 'chatgpt5' else config.generator_model %}
  <h1>Qasper RAG Test Results</h1>
  <div class="meta">
    <p>Results file: <code>{{ results_path }}</code></p>
    <p>
      <strong>Generator:</strong> <code id="config-generator" class="config-value">{{ generator_label }}</code>
      &nbsp;|&nbsp; <strong>Chunk size:</strong> <code id="config-chunk_size" class="config-value">{{ config.chunk_size }}</code>
      &nbsp;|&nbsp; <strong>Retrieval k:</strong> <code id="config-retrieval_k" class="config-value">{{ config.retrieval_k }}</code>
      &nbsp;|&nbsp; <strong>Articles:</strong> <code id="config-articles" class="config-value">{{ config.articles }}</code>
      &nbsp;|&nbsp; <strong>Questions/article:</strong> <code id="config-questions_per_article" class="config-value">{{ config.questions_per_article }}</code>
      &nbsp;|&nbsp; <strong>Split:</strong> <code id="config-split" class="config-value">{{ config.split }}</code>
      &nbsp;|&nbsp; <strong>Datasets:</strong> <code id="config-dataset" class="config-value">{{ config.datasets or config.dataset }}</code>
      &nbsp;|&nbsp; <strong>Show context:</strong> <code id="config-show_context" class="config-value">{{ 'Yes' if config.show_context else 'No' }}</code>
    </p>
    <p>Total questions processed: <strong id="total-questions">0</strong></p>
  </div>
  <div class="config-form" id="config-form">
    <label style="display: flex; flex-direction: column; gap: 0.5rem;">
      <span style="font-size: 0.95rem; color: #495057; font-weight: 500;">Datasets</span>
      <div style="display: flex; flex-direction: column; gap: 0.4rem;">
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-dataset-qasper" name="datasets" value="qasper" class="dataset-checkbox" {% if config.dataset == 'qasper' or (config.datasets and 'qasper' in config.datasets) %}checked{% endif %}>
          <span>QASPER</span>
        </label>
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-dataset-qmsum" name="datasets" value="qmsum" class="dataset-checkbox" {% if config.dataset == 'qmsum' or (config.datasets and 'qmsum' in config.datasets) %}checked{% endif %}>
          <span>QMSum</span>
        </label>
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-dataset-narrativeqa" name="datasets" value="narrativeqa" class="dataset-checkbox" {% if config.dataset == 'narrativeqa' or (config.datasets and 'narrativeqa' in config.datasets) %}checked{% endif %}>
          <span>NarrativeQA</span>
        </label>
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-dataset-quality" name="datasets" value="quality" class="dataset-checkbox" {% if config.dataset == 'quality' or (config.datasets and 'quality' in config.datasets) %}checked{% endif %}>
          <span>QuALITY</span>
        </label>
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-dataset-hotpot" name="datasets" value="hotpot" class="dataset-checkbox" {% if config.dataset == 'hotpot' or (config.datasets and 'hotpot' in config.datasets) %}checked{% endif %}>
          <span>HotpotQA</span>
        </label>
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-dataset-musique" name="datasets" value="musique" class="dataset-checkbox" {% if config.dataset == 'musique' or (config.datasets and 'musique' in config.datasets) %}checked{% endif %}>
          <span>MuSiQue</span>
        </label>
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-dataset-xsum" name="datasets" value="xsum" class="dataset-checkbox" {% if config.dataset == 'xsum' or (config.datasets and 'xsum' in config.datasets) %}checked{% endif %}>
          <span>XSum</span>
        </label>
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-dataset-wikiasp" name="datasets" value="wikiasp" class="dataset-checkbox" {% if config.dataset == 'wikiasp' or (config.datasets and 'wikiasp' in config.datasets) %}checked{% endif %}>
          <span>WikiAsp</span>
        </label>
        <label style="flex-direction: row; align-items: center; gap: 0.5rem; cursor: pointer;">
          <input type="checkbox" id="input-dataset-longbench" name="datasets" value="longbench" class="dataset-checkbox" {% if config.dataset == 'longbench' or (config.datasets and 'longbench' in config.datasets) %}checked{% endif %}>
          <span>LongBench</span>
        </label>
      </div>
    </label>
    <label for="input-generator_model">
      Generator
      <select id="input-generator_model" name="generator_model" class="param-input" data-default="{{ config.generator_model }}" data-type="string" data-target="config-generator">
        <option value="t5-small" {% if config.generator_model == 't5-small' %}selected{% endif %}>Local T5 Small</option>
        <option value="chatgpt5" {% if config.generator_model == 'chatgpt5' %}selected{% endif %}>ChatGPT 5</option>
      </select>
    </label>
    <label for="input-retrieval_k">
      Retrieval k
      <input id="input-retrieval_k" type="number" min="1" step="1" name="retrieval_k" class="param-input" value="{{ config.retrieval_k }}" data-default="{{ config.retrieval_k }}" data-type="number" data-target="config-retrieval_k">
    </label>
    <label for="input-chunk_size">
      Chunk size
      <input id="input-chunk_size" type="number" min="1" step="1" name="chunk_size" class="param-input" value="{{ config.chunk_size }}" data-default="{{ config.chunk_size }}" data-type="number" data-target="config-chunk_size">
    </label>
    <label for="input-articles">
      Articles
      <input id="input-articles" type="number" min="1" step="1" name="articles" class="param-input" value="{{ config.articles }}" data-default="{{ config.articles }}" data-type="number" data-target="config-articles">
    </label>
    <label for="input-questions_per_article">
      Questions/article
      <input id="input-questions_per_article" type="number" min="0" step="1" name="questions_per_article" class="param-input" value="{{ config.questions_per_article }}" data-default="{{ config.questions_per_article }}" data-type="number" data-target="config-questions_per_article">
    </label>
    <label id="chatgpt5-key-field" for="input-chatgpt5_api_key" style="display: {{ 'flex' if config.generator_model == 'chatgpt5' else 'none' }};">
      ChatGPT5 API key
      <div class="api-key-actions">
        <input id="input-chatgpt5_api_key" type="password" class="api-key-input" placeholder="Enter API key" autocomplete="off">
        <button type="button" id="validate-api-key">Validate</button>
        <span id="api-key-status" class="api-key-status"></span>
      </div>
    </label>
    <label for="input-show_context" style="flex-direction: row; align-items: center; gap: 0.5rem;">
      <input id="input-show_context" type="checkbox" name="show_context" class="param-input" data-default="{{ 'true' if config.show_context else 'false' }}" data-type="boolean" data-target="config-show_context" {% if config.show_context %}checked{% endif %}>
      Show retrieved context
    </label>
  </div>
  <div class="actions">
    <button id="run-tests" type="button" {% if run_state.status == 'running' %}disabled{% endif %}>Run Tests</button>
    <span id="run-status" data-status="{{ run_state.status }}" data-last-result="{{ run_state.last_result or '' }}">{{ run_state.message }}</span>
  </div>
  <div id="dataset-tabs-container"></div>
  <div id="empty-state-container" style="text-align: center; padding: 2rem; color: #6c757d;">
    <p>No results yet. Click <strong>Run Tests</strong> to begin.</p>
  </div>
  <script>
    console.log('JavaScript script loading...');
    const runButton = document.getElementById('run-tests');
    const runStatus = document.getElementById('run-status');
    const datasetTabsContainer = document.getElementById('dataset-tabs-container');
    const emptyStateContainer = document.getElementById('empty-state-container');
    const totalQuestions = document.getElementById('total-questions');
    const generatorSelect = document.getElementById('input-generator_model');
    const apiKeyField = document.getElementById('chatgpt5-key-field');
    const apiKeyInput = document.getElementById('input-chatgpt5_api_key');
    const validateButton = document.getElementById('validate-api-key');
    const apiKeyStatus = document.getElementById('api-key-status');
    const paramInputs = Array.from(document.querySelectorAll('.param-input'));
    const datasetCheckboxes = Array.from(document.querySelectorAll('.dataset-checkbox'));
    let renderedCount = 0;
    
    console.log('Button element:', runButton);
    console.log('Button exists:', !!runButton);
    if (runButton) {
      console.log('Button type:', runButton.type);
      console.log('Button disabled:', runButton.disabled);
    }

    function updateDatasetDisplay() {
      const checked = Array.from(datasetCheckboxes)
        .filter(cb => cb.checked)
        .map(cb => cb.value);
      const target = document.getElementById('config-dataset');
      if (target) {
        const displayText = checked.map(d => {
          if (d === 'qasper') return 'QASPER';
          if (d === 'qmsum') return 'QMSum';
          if (d === 'narrativeqa') return 'NarrativeQA';
          if (d === 'quality') return 'QuALITY';
          if (d === 'hotpot') return 'HotpotQA';
          if (d === 'musique') return 'MuSiQue';
          if (d === 'xsum') return 'XSum';
          if (d === 'wikiasp') return 'WikiAsp';
          if (d === 'longbench') return 'LongBench';
          return d;
        }).join(', ') || 'None';
        target.textContent = displayText;
        const defaultDatasets = Array.from(datasetCheckboxes)
          .filter(cb => cb.hasAttribute('checked'))
          .map(cb => cb.value);
        const isModified = JSON.stringify(checked.sort()) !== JSON.stringify(defaultDatasets.sort());
        target.classList.toggle('modified', isModified);
      }
    }

    datasetCheckboxes.forEach((checkbox) => {
      checkbox.addEventListener('change', updateDatasetDisplay);
    });
    updateDatasetDisplay();

    function applyInputAppearance(input) {
      const defaultValue = input.dataset.default ?? '';
      const targetId = input.dataset.target;
      const target = targetId ? document.getElementById(targetId) : null;
      const type = input.dataset.type || input.type || 'text';
      let value = input.value.trim();
      if (type === 'boolean' && input instanceof HTMLInputElement) {
        value = input.checked ? 'true' : 'false';
      }
      const isModified = value !== '' && value !== defaultValue;
      input.classList.toggle('modified', isModified);
      if (target) {
        let displayValue = value === '' ? defaultValue : value;
        if (targetId === 'config-generator') {
          if (displayValue === '') {
            displayValue = defaultValue;
          }
          if (displayValue === 'chatgpt5') {
            displayValue = 'ChatGPT 5';
          }
        } else if (targetId === 'config-show_context') {
          displayValue = displayValue === 'true' ? 'Yes' : 'No';
        } else if (targetId === 'config-dataset') {
          // Display selected datasets
            if (Array.isArray(displayValue)) {
            displayValue = displayValue.map(d => {
              if (d === 'qasper') return 'QASPER';
              if (d === 'qmsum') return 'QMSum';
              if (d === 'narrativeqa') return 'NarrativeQA';
              if (d === 'quality') return 'QuALITY';
              if (d === 'hotpot') return 'HotpotQA';
              if (d === 'musique') return 'MuSiQue';
              return d;
            }).join(', ');
          } else {
            // Fallback for single dataset
            if (displayValue === 'qasper') {
              displayValue = 'QASPER';
            } else if (displayValue === 'qmsum') {
              displayValue = 'QMSum';
            } else if (displayValue === 'narrativeqa') {
              displayValue = 'NarrativeQA';
            } else if (displayValue === 'quality') {
              displayValue = 'QuALITY';
            } else if (displayValue === 'hotpot') {
              displayValue = 'HotpotQA';
            } else if (displayValue === 'musique') {
              displayValue = 'MuSiQue';
            } else if (displayValue === 'xsum') {
              displayValue = 'XSum';
            } else if (displayValue === 'wikiasp') {
              displayValue = 'WikiAsp';
            } else if (displayValue === 'longbench') {
              displayValue = 'LongBench';
            }
          }
        }
        target.textContent = displayValue;
        target.classList.toggle('modified', isModified);
      }
    }

    paramInputs.forEach((input) => {
      applyInputAppearance(input);
      const handler = () => applyInputAppearance(input);
      input.addEventListener('input', handler);
      if (input.tagName === 'SELECT' || input.dataset.type === 'boolean') {
        input.addEventListener('change', handler);
      }
    });

    function setApiKeyStatus(text, className) {
      if (!apiKeyStatus) {
        return;
      }
      apiKeyStatus.textContent = text;
      apiKeyStatus.className = className || 'api-key-status';
    }

    function validateApiKey() {
      if (!apiKeyInput) {
        return;
      }
      const key = apiKeyInput.value.trim();
      if (!key) {
        setApiKeyStatus('Enter a key to validate.', 'api-key-status invalid');
        return;
      }
      setApiKeyStatus('Validating...', 'api-key-status');
      if (validateButton) {
        validateButton.disabled = true;
      }
      fetch('/validate-chatgpt5', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: key }),
      })
        .then(async (res) => {
          const data = await res.json();
          if (!res.ok || !data.success) {
            throw new Error(data.message || 'Invalid API key');
          }
          setApiKeyStatus('API key is valid.', 'api-key-status valid');
        })
        .catch((err) => {
          setApiKeyStatus(err.message || 'Validation failed.', 'api-key-status invalid');
        })
        .finally(() => {
          if (validateButton) {
            validateButton.disabled = generatorSelect && generatorSelect.value !== 'chatgpt5';
          }
        });
    }

    if (validateButton) {
      validateButton.addEventListener('click', validateApiKey);
    }

    if (apiKeyInput) {
      apiKeyInput.addEventListener('input', () => setApiKeyStatus('', 'api-key-status'));
    }

    function updateGeneratorState() {
      if (!generatorSelect || !apiKeyField) {
        return;
      }
      const useChatGPT = generatorSelect.value === 'chatgpt5';
      apiKeyField.style.display = useChatGPT ? 'flex' : 'none';
      if (!useChatGPT) {
        if (apiKeyInput) {
          apiKeyInput.value = '';
        }
        if (apiKeyStatus) {
          apiKeyStatus.textContent = '';
          apiKeyStatus.className = 'api-key-status';
        }
      }
      if (validateButton) {
        validateButton.disabled = !useChatGPT;
      }
      applyInputAppearance(generatorSelect);
    }

    if (generatorSelect) {
      updateGeneratorState();
      generatorSelect.addEventListener('change', updateGeneratorState);
    }

    function updateStatus(data) {
      runButton.disabled = data.status === 'running';
      runStatus.textContent = data.message || '';
      runStatus.dataset.status = data.status || '';
      runStatus.dataset.lastResult = data.last_result || '';
    }

    function pollStatus() {
      fetch('/api/status')
        .then((res) => res.json())
        .then(updateStatus)
        .catch((err) => console.error('Status poll failed', err));
    }


    function formatMetricValue(value) {
      if (value === null || value === undefined || isNaN(value)) {
        return 'N/A';
      }
      return value.toFixed(4);
    }

    function formatMetricShort(value) {
      if (value === null || value === undefined || isNaN(value)) {
        return 'N/A';
      }
      return value.toFixed(3);
    }

    let renderedResultsByDataset = {};
    const tabExpandedState = {}; // Track expanded/collapsed state for each dataset

    function appendResultRowToTable(tbody, item) {
      const row = document.createElement('tr');
      const articleCell = document.createElement('td');
      articleCell.innerHTML = '<div class="badge">ID: ' + (item.article_id || 'N/A') + '</div><div>' + (item.title || 'Unknown title') + '</div>';
      const questionCell = document.createElement('td');
      questionCell.className = 'question';
      questionCell.textContent = item.question || '';
      const answerCell = document.createElement('td');
      answerCell.className = 'answer';
      const answerText = document.createElement('div');
      answerText.textContent = item.generated_answer || '';
      answerCell.appendChild(answerText);
      const showContext = document.getElementById('input-show_context')?.checked !== false;
      if (Array.isArray(item.retrieved_context) && item.retrieved_context.length > 0 && showContext) {
        const contextDiv = document.createElement('div');
        contextDiv.className = 'context';
        const contextTitle = document.createElement('strong');
        contextTitle.textContent = 'Retrieved Context:';
        contextDiv.appendChild(contextTitle);
        const maxPreview = Math.min(item.retrieved_context.length, 3);
        for (let i = 0; i < maxPreview; i++) {
          const ctx = item.retrieved_context[i] || {};
          const ctxDiv = document.createElement('div');
          ctxDiv.className = 'context-item';
          const section = ctx.section_title || 'Unknown Section';
          const para = ctx.paragraph_index !== undefined && ctx.paragraph_index !== null ? ctx.paragraph_index : '?';
          ctxDiv.innerHTML = '<em>' + section + ' ¶' + para + '</em><br>' + (ctx.text_preview || (ctx.chunk ? ctx.chunk.slice(0, 200) + (ctx.chunk.length > 200 ? '...' : '') : ''));
          contextDiv.appendChild(ctxDiv);
        }
        if (item.retrieved_context.length > 3) {
          const moreDiv = document.createElement('div');
          moreDiv.className = 'context-item';
          moreDiv.innerHTML = '<em>+ ' + (item.retrieved_context.length - 3) + ' more chunks</em>';
          contextDiv.appendChild(moreDiv);
        }
        answerCell.appendChild(contextDiv);
      }
      const refCell = document.createElement('td');
      if (Array.isArray(item.reference_answers) && item.reference_answers.length > 0) {
        const list = document.createElement('ul');
        for (const ref of item.reference_answers) {
          const li = document.createElement('li');
          li.textContent = ref;
          list.appendChild(li);
        }
        refCell.appendChild(list);
      } else {
        const placeholder = document.createElement('em');
        placeholder.textContent = 'No reference answers provided.';
        refCell.appendChild(placeholder);
      }
      const scoreCell = document.createElement('td');
      const scoreParts = [];
      if (item.exact_match !== null && item.exact_match !== undefined) {
        scoreParts.push('<strong>EM:</strong> ' + formatMetricValue(item.exact_match));
      }
      if (item.f1_score !== null && item.f1_score !== undefined) {
        scoreParts.push('<strong>F1:</strong> ' + formatMetricValue(item.f1_score));
      }
      if (item.recall_score !== null && item.recall_score !== undefined) {
        scoreParts.push('<strong>Recall:</strong> ' + formatMetricValue(item.recall_score));
      }
      if (item.rouge_l_score !== null && item.rouge_l_score !== undefined) {
        scoreParts.push('<strong>R-L:</strong> ' + formatMetricValue(item.rouge_l_score));
      }
      if (item.bleu_score !== null && item.bleu_score !== undefined) {
        scoreParts.push('<strong>BLEU:</strong> ' + formatMetricValue(item.bleu_score));
      }
      scoreCell.innerHTML = scoreParts.length > 0 ? scoreParts.join('<br>') : '<em>No scores</em>';
      row.appendChild(articleCell);
      row.appendChild(questionCell);
      row.appendChild(answerCell);
      row.appendChild(refCell);
      row.appendChild(scoreCell);
      tbody.appendChild(row);
    }

    function renderDatasetTabs(allResults, statsData) {
      if (!datasetTabsContainer) {
        return;
      }

      // Group results by dataset
      const resultsByDataset = {};
      allResults.forEach((result) => {
        const dataset = result.dataset || 'unknown';
        if (!resultsByDataset[dataset]) {
          resultsByDataset[dataset] = [];
        }
        resultsByDataset[dataset].push(result);
      });

      const datasets = Object.keys(resultsByDataset);
      if (datasets.length === 0) {
        datasetTabsContainer.innerHTML = '';
        if (emptyStateContainer) {
          emptyStateContainer.style.display = 'block';
        }
        return;
      }

      if (emptyStateContainer) {
        emptyStateContainer.style.display = 'none';
      }

      // Preserve expanded state before clearing
      const existingTabs = datasetTabsContainer.querySelectorAll('.dataset-tab');
      existingTabs.forEach((tab) => {
        const datasetId = tab.id.replace('tab-', '');
        const header = tab.querySelector('.tab-header');
        const content = tab.querySelector('.tab-content');
        if (header && content) {
          const isExpanded = content.classList.contains('expanded');
          if (isExpanded) {
            tabExpandedState[datasetId] = true;
          } else if (tabExpandedState[datasetId] === undefined) {
            // Only set to false if not previously tracked
            tabExpandedState[datasetId] = false;
          }
        }
      });

      datasetTabsContainer.innerHTML = '';

      datasets.forEach((dataset) => {
        // Restore expanded state if previously expanded
        const shouldBeExpanded = tabExpandedState[dataset] === true;
        const datasetResults = resultsByDataset[dataset] || [];
        const stats = statsData[dataset] || { count: 0, metrics: {} };
        const datasetLabel = {
          'qasper': 'QASPER',
          'qmsum': 'QMSum',
          'narrativeqa': 'NarrativeQA',
          'quality': 'QuALITY',
          'hotpot': 'HotpotQA',
          'musique': 'MuSiQue',
          'xsum': 'XSum',
          'wikiasp': 'WikiAsp',
          'longbench': 'LongBench'
        }[dataset] || dataset;

        const tab = document.createElement('div');
        tab.className = 'dataset-tab';
        tab.id = 'tab-' + dataset;

        const header = document.createElement('div');
        header.className = shouldBeExpanded ? 'tab-header' : 'tab-header collapsed';

        const headerRow = document.createElement('div');
        headerRow.className = 'tab-header-row';

        const title = document.createElement('div');
        title.className = 'tab-title';
        let titleText = datasetLabel;
        const questionsTested = stats.count || 0;
        const questionsAvailable = stats.total_questions_available;
        const questionsInDataset = stats.total_questions_in_dataset;
        
        // Build title with question counts
        let countText = '';
        if (questionsInDataset !== null && questionsInDataset !== undefined) {
          // Show: "X of Y tested (Z total in dataset)"
          if (questionsAvailable !== null && questionsAvailable !== undefined) {
            countText = questionsTested + ' of ' + questionsAvailable + ' tested (' + questionsInDataset + ' total in dataset)';
          } else {
            countText = questionsTested + ' tested (' + questionsInDataset + ' total in dataset)';
          }
        } else if (questionsAvailable !== null && questionsAvailable !== undefined) {
          // Fallback: show available if we don't have total in dataset
          countText = questionsTested + ' of ' + questionsAvailable + ' tested';
        } else {
          // Last fallback: just show tested
          countText = questionsTested + ' questions tested';
        }
        titleText += ' (' + countText + ')';
        title.textContent = titleText;

        const toggle = document.createElement('span');
        toggle.className = 'tab-toggle';
        toggle.textContent = '▼';

        headerRow.appendChild(title);
        headerRow.appendChild(toggle);

        const statsRow = document.createElement('div');
        statsRow.className = 'tab-stats';
        
        const metrics = [
          { key: 'exact_match', label: 'EM' },
          { key: 'f1_score', label: 'F1' },
          { key: 'recall_score', label: 'Recall' },
          { key: 'rouge_l_score', label: 'ROUGE-L' },
          { key: 'bleu_score', label: 'BLEU' }
        ];

        metrics.forEach((metric) => {
          const metricStat = stats.metrics[metric.key] || {};
          const avg = metricStat.average;
          if (avg !== null && avg !== undefined && !isNaN(avg)) {
            const statItem = document.createElement('div');
            statItem.className = 'tab-stat-item';
            const label = document.createElement('span');
            label.className = 'tab-stat-label';
            label.textContent = metric.label + ':';
            const value = document.createElement('span');
            value.className = 'tab-stat-value';
            value.textContent = formatMetricShort(avg);
            statItem.appendChild(label);
            statItem.appendChild(value);
            statsRow.appendChild(statItem);
          }
        });

        header.appendChild(headerRow);
        header.appendChild(statsRow);

        const content = document.createElement('div');
        // Restore expanded state if previously expanded
        content.className = shouldBeExpanded ? 'tab-content expanded' : 'tab-content';

        if (datasetResults.length === 0) {
          const emptyState = document.createElement('div');
          emptyState.className = 'tab-empty-state';
          emptyState.textContent = 'No results for this dataset.';
          content.appendChild(emptyState);
        } else {
          const table = document.createElement('table');
          table.className = 'tab-results-table';
          
          const thead = document.createElement('thead');
          const headerRow = document.createElement('tr');
          const headers = ['Article', 'Question', 'Generated Answer', 'Reference Answers', 'Score'];
          headers.forEach(h => {
            const th = document.createElement('th');
            th.textContent = h;
            headerRow.appendChild(th);
          });
          thead.appendChild(headerRow);
          table.appendChild(thead);

          const tbody = document.createElement('tbody');
          // Render all results for this dataset
          datasetResults.forEach((result) => {
            appendResultRowToTable(tbody, result);
          });

          table.appendChild(tbody);
          content.appendChild(table);
        }

        header.onclick = function(e) {
          e.preventDefault();
          e.stopPropagation();
          const isCollapsed = header.classList.contains('collapsed');
          if (isCollapsed) {
            // Expand: remove collapsed, add expanded
            header.classList.remove('collapsed');
            content.classList.add('expanded');
            tabExpandedState[dataset] = true;
          } else {
            // Collapse: add collapsed, remove expanded
            header.classList.add('collapsed');
            content.classList.remove('expanded');
            tabExpandedState[dataset] = false;
          }
        };

        tab.appendChild(header);
        tab.appendChild(content);
        datasetTabsContainer.appendChild(tab);
      });
    }

    function fetchResults() {
      fetch('/api/results')
        .then((res) => res.json())
        .then((data) => {
          if (!Array.isArray(data)) {
            return;
          }
          if (totalQuestions) {
            totalQuestions.textContent = String(data.length);
          }
          
          // Fetch statistics and render tabs
          fetch('/api/statistics')
            .then((res) => res.json())
            .then((stats) => {
              renderDatasetTabs(data, stats);
            })
            .catch((err) => {
              console.error('Failed to fetch statistics', err);
              renderDatasetTabs(data, {});
            });
        })
        .catch((err) => console.error('Failed to fetch results', err));
    }

    function resetResultsDisplay() {
      renderedResultsByDataset = {};
      if (datasetTabsContainer) {
        datasetTabsContainer.innerHTML = '';
      }
      if (emptyStateContainer) {
        emptyStateContainer.style.display = 'block';
        emptyStateContainer.textContent = 'Running tests... results will appear here.';
      }
      if (totalQuestions) {
        totalQuestions.textContent = '0';
      }
    }

    console.log('Setting up event listeners...');
    console.log('runButton found:', !!runButton);
    
    try {
      if (runButton) {
        console.log('Attaching click event listener to run button');
        runButton.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        console.log('Run Tests button clicked');
        runButton.disabled = true;
        runStatus.textContent = 'Starting tests...';
        resetResultsDisplay();
        const overrides = {};
        paramInputs.forEach((input) => {
          const name = input.name;
          if (!name) {
            return;
          }
          if (name === 'generator_model') {
            const defaultValue = input.dataset.default ?? '';
            const value = input.value.trim();
            if (value !== '' && value !== defaultValue) {
              overrides.generator_model = value;
            }
            return;
          }
          const defaultValue = input.dataset.default ?? '';
          const type = input.dataset.type || input.type || 'text';
          let value = input.value.trim();
          if (type === 'boolean' && input instanceof HTMLInputElement) {
            value = input.checked ? 'true' : 'false';
          }
          if (value === '' || value === defaultValue) {
            return;
          }
          if (type === 'number') {
            const parsedValue = Number(value);
            if (!Number.isNaN(parsedValue)) {
              overrides[name] = parsedValue;
            }
          } else if (type === 'boolean') {
            overrides[name] = value === 'true';
          } else {
            overrides[name] = value;
          }
        });

        const generatorDefault = generatorSelect ? (generatorSelect.dataset.default ?? '') : '';
        const generatorValue = generatorSelect ? generatorSelect.value.trim() : generatorDefault;
        if (generatorSelect && generatorValue !== generatorDefault) {
          overrides.generator_model = generatorValue;
        }
        
        // Handle multiple dataset selection from checkboxes
        const checkedDatasets = Array.from(datasetCheckboxes)
          .filter(cb => cb.checked)
          .map(cb => cb.value);
        
        if (checkedDatasets.length === 0) {
          runStatus.textContent = 'Please select at least one dataset.';
          runButton.disabled = false;
          return;
        }
        
        if (checkedDatasets.length === 1) {
          // Single dataset - use legacy 'dataset' field for backward compatibility
          overrides.dataset = checkedDatasets[0];
        } else {
          // Multiple datasets - use 'datasets' array
          overrides.datasets = checkedDatasets;
        }
        if (generatorValue === 'chatgpt5') {
          const apiKey = apiKeyInput ? apiKeyInput.value.trim() : '';
          if (!apiKey) {
            runStatus.textContent = 'Please enter a ChatGPT5 API key.';
            runButton.disabled = false;
            return;
          }
          if (!(apiKeyStatus && apiKeyStatus.classList.contains('valid'))) {
            runStatus.textContent = 'Please validate the ChatGPT5 API key before running tests.';
            runButton.disabled = false;
            return;
          }
          overrides.chatgpt5_api_key = apiKey;
        }
        
        // Add split parameter from the config display (it should be available)
        const splitDisplay = document.getElementById('config-split');
        if (splitDisplay && splitDisplay.textContent) {
          overrides.split = splitDisplay.textContent.trim();
        }
        
        console.log('Sending test run request with overrides:', { ...overrides, chatgpt5_api_key: overrides.chatgpt5_api_key ? '***' : undefined });
        
        fetch('/run-tests', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(overrides),
        })
          .then(async (res) => {
            console.log('Response status:', res.status, res.statusText);
            const data = await res.json();
            console.log('Response data:', data);
            if (!res.ok) {
              throw new Error(data.message || 'Failed to start tests');
            }
            updateStatus(data);
            fetchResults();
          })
          .catch((err) => {
            console.error('Fetch error:', err);
            runStatus.textContent = 'Error: ' + (err.message || 'Failed to start tests. Check console for details.');
            runButton.disabled = false;
          });
        });
        
        console.log('Event listener attached successfully');
      } else {
        console.error('runButton element not found! Cannot attach event listener.');
      }
    } catch (err) {
      console.error('Error setting up event listeners:', err);
      if (runStatus) {
        runStatus.textContent = 'Error: Failed to set up button. Check console for details.';
      }
    }

    try {
      setInterval(() => {
        pollStatus();
        fetchResults();
      }, 3000);
      fetchResults();
      console.log('Polling interval set up successfully');
    } catch (err) {
      console.error('Error setting up polling:', err);
    }
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    default_dataset = getattr(TEST_OPTIONS, "dataset", "qasper")
    default_datasets = getattr(TEST_OPTIONS, "datasets", None)
    return render_template_string(
        TEMPLATE,
        results_path=RESULTS_PATH,
        run_state=RUN_STATE,
        config={
            "generator_model": TEST_OPTIONS.generator_model,
            "chunk_size": TEST_OPTIONS.chunk_size,
            "retrieval_k": TEST_OPTIONS.retrieval_k,
            "articles": TEST_OPTIONS.articles,
            "questions_per_article": TEST_OPTIONS.questions_per_article,
            "split": TEST_OPTIONS.split,
            "dataset": default_dataset,  # For backward compatibility
            "datasets": default_datasets if default_datasets else [default_dataset],
            "show_context": TEST_OPTIONS.show_context,
        },
    )


@app.route("/api/status")
def api_status():
    return jsonify(RUN_STATE)


@app.route("/validate-chatgpt5", methods=["POST"])
def validate_chatgpt5():
    payload = request.get_json(silent=True) or {}
    api_key = payload.get("api_key")
    if not api_key:
        return jsonify({"success": False, "message": "API key is required."}), 400

    options_dict = vars(TEST_OPTIONS).copy()
    options_dict["chatgpt5_api_key"] = api_key
    options_dict["generator_model"] = "chatgpt5"
    temp_options = argparse.Namespace(**options_dict)

    clear_results_file()
    try:
        run_test_script(temp_options)
    except Exception as exc:
        app.logger.error("API key validation failed: %s", exc)
        return jsonify({"success": False, "message": str(exc)}), 500

    results = load_results()
    is_valid = bool(results)
    message = "API key is valid." if is_valid else "Unable to confirm API key validity."
    return jsonify({"success": is_valid, "message": message})


@app.route("/api/results")
def api_results():
    return jsonify(load_results())


@app.route("/api/statistics")
def api_statistics():
    """Return statistics for each dataset."""
    results = load_results()
    stats = calculate_dataset_statistics(results)
    return jsonify(stats)


@app.route("/run-tests", methods=["POST"])
def trigger_run():
    app.logger.info("Received POST request to /run-tests")
    payload = request.get_json(silent=True) or {}
    app.logger.info("Request payload (without API key): %s", {k: v for k, v in payload.items() if k != "chatgpt5_api_key"})
    options_dict = vars(TEST_OPTIONS).copy()

    for field in ("retrieval_k", "chunk_size", "questions_per_article", "articles"):
        if field in payload:
            try:
                options_dict[field] = int(payload[field])
            except (TypeError, ValueError):
                app.logger.warning("Ignoring invalid override for %s: %s", field, payload[field])

    if "show_context" in payload:
        options_dict["show_context"] = bool(payload["show_context"])

    if "split" in payload and payload["split"]:
        options_dict["split"] = str(payload["split"]).strip()

    if "generator_model" in payload and payload["generator_model"]:
        options_dict["generator_model"] = str(payload["generator_model"]).strip()

    # Handle dataset selection (single or multiple)
    if "datasets" in payload and isinstance(payload["datasets"], list) and payload["datasets"]:
        # Multiple datasets
        datasets = [str(d).strip().lower() for d in payload["datasets"]]
        valid_datasets = [d for d in datasets if d in ("qasper", "qmsum", "narrativeqa", "quality", "hotpot", "musique", "xsum", "wikiasp", "longbench")]
        if valid_datasets:
            if len(valid_datasets) == 1:
                # Single dataset - use legacy field for compatibility
                options_dict["dataset"] = valid_datasets[0]
            else:
                # Multiple datasets
                options_dict["datasets"] = valid_datasets
                options_dict["dataset"] = valid_datasets[0]  # Keep for backward compatibility
        else:
            options_dict["dataset"] = "qasper"
    elif "dataset" in payload and payload["dataset"]:
        # Single dataset (legacy support)
        options_dict["dataset"] = str(payload["dataset"]).strip().lower()
        if options_dict["dataset"] not in ("qasper", "qmsum", "narrativeqa", "quality", "hotpot", "musique", "xsum", "wikiasp", "longbench"):
            options_dict["dataset"] = "qasper"

    generator_value = str(options_dict.get("generator_model", TEST_OPTIONS.generator_model or "t5-small")).lower()
    api_key_override = payload.get("chatgpt5_api_key")

    if generator_value == "chatgpt5":
        if not api_key_override:
            return jsonify({"status": "error", "message": "ChatGPT5 generator requires an API key."}), 400
        options_dict["chatgpt5_api_key"] = api_key_override
    else:
        options_dict["chatgpt5_api_key"] = None

    updated_options = argparse.Namespace(**options_dict)
    
    # Log the options being used
    app.logger.info("Triggering test run with options: %s", {k: v for k, v in options_dict.items() if k != "chatgpt5_api_key"})

    clear_results_file()
    RUN_STATE["message"] = "Test run initiated..."
    if not start_test_run(updated_options, async_run=True):
        app.logger.warning("Failed to start test run: lock already held")
        return jsonify({"status": "busy", "message": "A run is already in progress."}), 409
    app.logger.info("Test run started successfully")
    return jsonify({"status": "running", "message": "Test run started..."})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the QASPER RAG interface.")
    parser.add_argument("--split", default="validation", help="Dataset split to evaluate.")
    parser.add_argument("--articles", type=int, default=1, help="Number of articles to evaluate (0 means all).")
    parser.add_argument(
        "--questions-per-article",
        type=int,
        default=3,
        help="Maximum questions per article (0 means all questions).",
    )
    parser.add_argument("--retrieval-k", type=int, default=5, help="Chunks to retrieve for each query.")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters for indexing.")
    parser.add_argument(
        "--generator-model",
        default="t5-small",
        help="Hugging Face model identifier for answer generation.",
    )
    parser.add_argument("--chatgpt5-api-key", default=None, help="API key for ChatGPT5 generator.")
    parser.add_argument("--show-context", dest="show_context", action="store_true", help="Display retrieved context in outputs.")
    parser.add_argument("--hide-context", dest="show_context", action="store_false", help="Hide retrieved context in outputs.")
    parser.set_defaults(show_context=False)
    parser.add_argument("--dataset", choices=["qasper", "qmsum", "narrativeqa", "quality", "hotpot", "musique", "xsum", "wikiasp", "longbench"], default="qasper", help="Dataset to use: 'qasper', 'qmsum', 'narrativeqa', 'quality', 'hotpot', 'musique', 'xsum', 'wikiasp', or 'longbench'.")
    parser.add_argument("--log-level", default="INFO", help="Log level passed to the test script.")
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH), help="Where to write the results JSON.")
    parser.add_argument("--host", default="0.0.0.0", help="Interface host.")
    parser.add_argument("--port", type=int, default=5051, help="Interface port.")
    parser.add_argument("--run-on-start", action="store_true", help="Run the test suite automatically at startup.")
    parser.add_argument("--no-open-browser", dest="open_browser", action="store_false", help="Do not open the interface in the default browser.")
    parser.set_defaults(open_browser=True)
    return parser.parse_args()


if __name__ == "__main__":
    TEST_OPTIONS = parse_args()
    RESULTS_PATH = Path(TEST_OPTIONS.results_path).resolve()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if TEST_OPTIONS.run_on_start:
        clear_results_file()
        start_test_run(TEST_OPTIONS, async_run=False)
    else:
        app.logger.info("Startup configured without automatic test run.")

    if TEST_OPTIONS.open_browser:
        target_host = TEST_OPTIONS.host if TEST_OPTIONS.host not in ("0.0.0.0", "::") else "127.0.0.1"
        Timer = threading.Timer
        Timer(1.5, lambda: webbrowser.open(f"http://{target_host}:{TEST_OPTIONS.port}")).start()

    app.run(host=TEST_OPTIONS.host, port=TEST_OPTIONS.port, debug=True)


