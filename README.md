# ADI Visit Structurer

![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![NLP](https://img.shields.io/badge/NLP-clinical--pipeline-orange)

A hybrid NLP pipeline for structuring home-care clinical visit notes into standardized JSON records.

This project was developed as part of an internship focused on building automated tools for processing and evaluating structured medical data extracted from narrative clinical notes.

---

## Overview

Clinical home-care visit notes often contain unstructured information such as:

- reason for visit
- vital signs
- follow-up instructions
- interventions
- clinical problems

This project automatically extracts and normalizes these elements into a structured format suitable for downstream analysis, reporting, and validation.

The system combines **rule-based extraction** with **LLM-assisted extraction**.

---

## Pipeline Architecture

The pipeline processes each visit note through several stages:

### 1. Preprocessing
- text normalization
- cleaning and segmentation

### 2. Rule-based extraction
- vital signs
- interventions
- reasons for visit

### 3. LLM extraction (Hybrid Mode)
- local LLM inference using **Llama 3 via Ollama**

### 4. Problem normalization
- mapping extracted conditions to normalized labels

### 5. Schema validation
- outputs validated against a JSON schema

### 6. Evaluation
- predictions compared with gold annotations

---

## Example Output

Each visit note is converted into structured JSON:

```json
{
  "clinical": {
    "reason_for_visit": "...",
    "follow_up": "...",
    "interventions": [],
    "vitals": {
      "blood_pressure_systolic": null,
      "blood_pressure_diastolic": null,
      "heart_rate": null,
      "temperature": null,
      "spo2": null
    }
  },
  "coding": {
    "problems_normalized": []
  }
}
````

---

## Project Structure

```
adi-visit-structurer
│
├── data
│   └── synthetic
│       ├── raw
│       ├── gold
│       └── pred
│
├── src
│   ├── run_pipeline.py
│   ├── extract_rules.py
│   ├── llm_extract.py
│   ├── normalize.py
│   ├── preprocess.py
│   ├── problem_evidence.py
│   ├── quality.py
│   ├── evaluate.py
│   └── export_reports.py
│
├── schemas
│   └── visit_schema_v1.json
│
├── reports
│   ├── dashboard.html
│   ├── metrics.json
│   └── summary_table.csv
│
├── tests
├── tools
├── notebooks
```

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

Run the extraction pipeline:

```bash
python -m src.run_pipeline --hybrid
```

Evaluate predictions:

```bash
python -m src.evaluate
```

---

## Reports

Evaluation results are exported to the `reports/` folder.

Key outputs include:

* `metrics.json`
* `summary_table.csv`
* `dashboard.html`

The dashboard provides a visual summary of pipeline performance.

---

## Technologies Used

* Python
* Rule-based NLP
* RapidFuzz
* JSON schema validation
* Ollama + Llama 3 (Hybrid extraction)
* CSV reporting
* HTML dashboard generation

---

## Author

**Maryam Amini**

Clinical NLP Pipeline – Internship Project
University of Messina