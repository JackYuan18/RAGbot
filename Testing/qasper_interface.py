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
from typing import Any, List

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
    if not DEFAULT_TEST_SCRIPT.exists():
        raise FileNotFoundError(f"Cannot locate test script at {DEFAULT_TEST_SCRIPT}")

    cmd = [
        sys.executable,
        str(DEFAULT_TEST_SCRIPT),
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
        str(RESULTS_PATH),
    ]

    if getattr(options, "chatgpt5_api_key", None):
        cmd.extend(["--chatgpt5-api-key", options.chatgpt5_api_key])

    if getattr(options, "show_context", True):
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
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def start_test_run(options: argparse.Namespace, *, async_run: bool = True) -> bool:
    if not RUN_LOCK.acquire(blocking=False):
        return False

    RUN_STATE["status"] = "running"
    RUN_STATE["message"] = "Test run in progress..."
    RUN_STATE["last_result"] = None

    def runner() -> None:
        try:
            result = run_test_script(options)
            if result.returncode == 0:
                RUN_STATE["message"] = "Last run completed successfully."
                RUN_STATE["last_result"] = "success"
                app.logger.info("test_qasper_rag.py completed successfully.")
            else:
                RUN_STATE["message"] = f"Run failed with code {result.returncode}."
                RUN_STATE["last_result"] = "error"
                app.logger.error("test_qasper_rag.py failed (%s):\n%s", result.returncode, result.stderr)
        except Exception as exc:  # pragma: no cover - defensive logging
            RUN_STATE["message"] = f"Error: {exc}"
            RUN_STATE["last_result"] = "error"
            app.logger.exception("Unexpected error during test run")
        finally:
            RUN_STATE["status"] = "idle"
            RUN_LOCK.release()

    if async_run:
        threading.Thread(target=runner, daemon=True).start()
    else:
        runner()

    return True


def load_results() -> List[dict[str, Any]]:
    if RESULTS_PATH.exists():
        try:
            with RESULTS_PATH.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    return data
        except Exception as exc:  # pragma: no cover - defensive logging
            app.logger.error("Failed to read %s: %s", RESULTS_PATH, exc)
    return []


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
      &nbsp;|&nbsp; <strong>Show context:</strong> <code id="config-show_context" class="config-value">{{ 'Yes' if config.show_context else 'No' }}</code>
    </p>
    <p>Total questions processed: <strong id="total-questions">0</strong></p>
  </div>
  <div class="config-form" id="config-form">
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
    <button id="run-tests" {% if run_state.status == 'running' %}disabled{% endif %}>Run Tests</button>
    <span id="run-status" data-status="{{ run_state.status }}" data-last-result="{{ run_state.last_result or '' }}">{{ run_state.message }}</span>
  </div>
  <div id="results-container">
    <p id="empty-state">No results yet. Click <strong>Run Tests</strong> to begin.</p>
    <table id="results-table" hidden>
    <thead>
      <tr>
        <th style="width: 15%">Article</th>
        <th style="width: 25%">Question</th>
        <th style="width: 25%">Generated Answer</th>
        <th style="width: 20%">Reference Answers</th>
        <th style="width: 15%">Score</th>
      </tr>
    </thead>
      <tbody id="results-body"></tbody>
    </table>
  </div>
  <script>
    const runButton = document.getElementById('run-tests');
    const runStatus = document.getElementById('run-status');
    const resultsTable = document.getElementById('results-table');
    const resultsBody = document.getElementById('results-body');
    const emptyState = document.getElementById('empty-state');
    const totalQuestions = document.getElementById('total-questions');
    const generatorSelect = document.getElementById('input-generator_model');
    const apiKeyField = document.getElementById('chatgpt5-key-field');
    const apiKeyInput = document.getElementById('input-chatgpt5_api_key');
    const validateButton = document.getElementById('validate-api-key');
    const apiKeyStatus = document.getElementById('api-key-status');
    const paramInputs = Array.from(document.querySelectorAll('.param-input'));
    let renderedCount = 0;

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

    function appendResultRow(item) {
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
      scoreCell.style.verticalAlign = 'top';
      scoreCell.style.fontSize = '0.9rem';
      
      const scoreContainer = document.createElement('div');
      scoreContainer.style.display = 'flex';
      scoreContainer.style.flexDirection = 'column';
      scoreContainer.style.gap = '0.5rem';
      
      let hasAnyScore = false;
      
      // Display Exact Match score
      if (item.exact_match !== null && item.exact_match !== undefined) {
        const emValue = parseFloat(item.exact_match);
        if (!isNaN(emValue)) {
          hasAnyScore = true;
          const emDiv = document.createElement('div');
          const emLabel = document.createElement('span');
          emLabel.textContent = 'EM: ';
          emLabel.style.fontWeight = '600';
          emLabel.style.color = '#495057';
          const emScore = document.createElement('span');
          emScore.textContent = emValue.toFixed(4);
          emScore.style.fontWeight = '600';
          emScore.style.color = emValue >= 1.0 ? '#198754' : '#dc3545';
          emDiv.appendChild(emLabel);
          emDiv.appendChild(emScore);
          scoreContainer.appendChild(emDiv);
        }
      }
      
      // Display F1 score
      if (item.f1_score !== null && item.f1_score !== undefined) {
        const f1Value = parseFloat(item.f1_score);
        if (!isNaN(f1Value)) {
          hasAnyScore = true;
          const f1Div = document.createElement('div');
          const f1Label = document.createElement('span');
          f1Label.textContent = 'F1: ';
          f1Label.style.fontWeight = '600';
          f1Label.style.color = '#495057';
          const f1Score = document.createElement('span');
          f1Score.textContent = f1Value.toFixed(4);
          f1Score.style.fontWeight = '600';
          // Color code based on F1 score: green for high (>0.5), yellow for medium (>0.2), red for low
          if (f1Value >= 0.5) {
            f1Score.style.color = '#198754';
          } else if (f1Value >= 0.2) {
            f1Score.style.color = '#ffc107';
          } else {
            f1Score.style.color = '#dc3545';
          }
          f1Div.appendChild(f1Label);
          f1Div.appendChild(f1Score);
          scoreContainer.appendChild(f1Div);
        }
      }
      
      // Display ROUGE-L score
      if (item.rouge_l_score !== null && item.rouge_l_score !== undefined) {
        const rougeLValue = parseFloat(item.rouge_l_score);
        if (!isNaN(rougeLValue)) {
          hasAnyScore = true;
          const rougeLDiv = document.createElement('div');
          const rougeLLabel = document.createElement('span');
          rougeLLabel.textContent = 'ROUGE-L: ';
          rougeLLabel.style.fontWeight = '600';
          rougeLLabel.style.color = '#495057';
          const rougeLScore = document.createElement('span');
          rougeLScore.textContent = rougeLValue.toFixed(4);
          rougeLScore.style.fontWeight = '600';
          // Color code based on ROUGE-L score: green for high (>0.5), yellow for medium (>0.2), red for low
          if (rougeLValue >= 0.5) {
            rougeLScore.style.color = '#198754';
          } else if (rougeLValue >= 0.2) {
            rougeLScore.style.color = '#ffc107';
          } else {
            rougeLScore.style.color = '#dc3545';
          }
          rougeLDiv.appendChild(rougeLLabel);
          rougeLDiv.appendChild(rougeLScore);
          scoreContainer.appendChild(rougeLDiv);
        }
      }
      
      // Display BLEU score
      if (item.bleu_score !== null && item.bleu_score !== undefined) {
        const bleuValue = parseFloat(item.bleu_score);
        if (!isNaN(bleuValue)) {
          hasAnyScore = true;
          const bleuDiv = document.createElement('div');
          const bleuLabel = document.createElement('span');
          bleuLabel.textContent = 'BLEU: ';
          bleuLabel.style.fontWeight = '600';
          bleuLabel.style.color = '#495057';
          const bleuScore = document.createElement('span');
          bleuScore.textContent = bleuValue.toFixed(4);
          bleuScore.style.fontWeight = '600';
          // Color code based on BLEU score: green for high (>0.5), yellow for medium (>0.2), red for low
          if (bleuValue >= 0.5) {
            bleuScore.style.color = '#198754';
          } else if (bleuValue >= 0.2) {
            bleuScore.style.color = '#ffc107';
          } else {
            bleuScore.style.color = '#dc3545';
          }
          bleuDiv.appendChild(bleuLabel);
          bleuDiv.appendChild(bleuScore);
          scoreContainer.appendChild(bleuDiv);
        }
      }
      
      if (hasAnyScore) {
        scoreCell.appendChild(scoreContainer);
      } else {
        scoreCell.innerHTML = '<em style="color: #6c757d;">N/A</em>';
      }

      row.appendChild(articleCell);
      row.appendChild(questionCell);
      row.appendChild(answerCell);
      row.appendChild(refCell);
      row.appendChild(scoreCell);
      resultsBody.appendChild(row);
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
          if (data.length > 0) {
            if (resultsTable.hidden) {
              resultsTable.hidden = false;
            }
            if (!emptyState.hidden) {
              emptyState.hidden = true;
            }
            for (; renderedCount < data.length; renderedCount += 1) {
              appendResultRow(data[renderedCount]);
            }
          } else if (renderedCount === 0) {
            resultsTable.hidden = true;
            emptyState.hidden = false;
          }
        })
        .catch((err) => console.error('Failed to fetch results', err));
    }

    function resetResultsDisplay() {
      renderedCount = 0;
      resultsBody.innerHTML = '';
      resultsTable.hidden = true;
      emptyState.hidden = false;
      emptyState.textContent = 'Running tests... results will appear here.';
      if (totalQuestions) {
        totalQuestions.textContent = '0';
      }
    }

    if (runButton) {
      runButton.addEventListener('click', () => {
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
        fetch('/run-tests', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(overrides),
        })
          .then(async (res) => {
            const data = await res.json();
            if (!res.ok) {
              throw new Error(data.message || 'Failed to start tests');
            }
            updateStatus(data);
            fetchResults();
          })
          .catch((err) => {
            runStatus.textContent = 'Error: ' + err.message;
            runButton.disabled = false;
          });
      });

      setInterval(() => {
        pollStatus();
        fetchResults();
      }, 3000);
      fetchResults();
    }
  </script>
</body>
</html>
"""


@app.route("/")
def index():
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


@app.route("/run-tests", methods=["POST"])
def trigger_run():
    payload = request.get_json(silent=True) or {}
    options_dict = vars(TEST_OPTIONS).copy()

    for field in ("retrieval_k", "chunk_size", "questions_per_article", "articles"):
        if field in payload:
            try:
                options_dict[field] = int(payload[field])
            except (TypeError, ValueError):
                app.logger.warning("Ignoring invalid override for %s: %s", field, payload[field])

    if "show_context" in payload:
        options_dict["show_context"] = bool(payload["show_context"])

    if "generator_model" in payload and payload["generator_model"]:
        options_dict["generator_model"] = str(payload["generator_model"]).strip()

    generator_value = str(options_dict.get("generator_model", TEST_OPTIONS.generator_model or "t5-small")).lower()
    api_key_override = payload.get("chatgpt5_api_key")

    if generator_value == "chatgpt5":
        if not api_key_override:
            return jsonify({"status": "error", "message": "ChatGPT5 generator requires an API key."}), 400
        options_dict["chatgpt5_api_key"] = api_key_override
    else:
        options_dict["chatgpt5_api_key"] = None

    updated_options = argparse.Namespace(**options_dict)

    clear_results_file()
    RUN_STATE["message"] = "Test run initiated..."
    if not start_test_run(updated_options, async_run=True):
        return jsonify({"status": "busy", "message": "A run is already in progress."}), 409
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
    parser.set_defaults(show_context=True)
    parser.add_argument("--log-level", default="INFO", help="Log level passed to the test script.")
    parser.add_argument("--results-path", default=str(DEFAULT_RESULTS_PATH), help="Where to write the results JSON.")
    parser.add_argument("--host", default="0.0.0.0", help="Interface host.")
    parser.add_argument("--port", type=int, default=5051, help="Interface port.")
    parser.add_argument("--run-on-start", action="store_true", help="Run the test suite automatically at startup.")
    parser.add_argument("--open-browser", action="store_true", help="Open the interface in the default browser.")
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


