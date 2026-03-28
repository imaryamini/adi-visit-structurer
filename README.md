# ADI Assistant

ADI Assistant is a prototype system for transforming dictated or typed home-care visit notes into structured ADI report drafts.

The project was developed in the context of an internship focused on ADI (Assistenza Domiciliare Integrata) documentation, with the goal of reducing manual form-filling effort by structuring free-text clinical dictations into a standardized output.

## Main Features

- Voice recording through the web interface
- Speech-to-text transcription
- Manual text input for clinical notes
- Rule-based clinical information extraction
- Structured ADI-style report generation
- Basic quality checks for missing mandatory fields
- Web interface with login, register, and assistant pages

## Project Structure

```text
adi-visit-structurer/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── templates/
│   ├── login.html
│   ├── register.html
│   └── index.html
├── static/
│   ├── style.css
│   └── app.js
├── src/
│   ├── voice_input.py
│   ├── run_pipeline.py
│   ├── preprocess.py
│   ├── extract_rules.py
│   ├── normalize.py
│   ├── quality.py
│   ├── schema.py
│   ├── problem_evidence.py
│   ├── llm_extract.py
│   ├── evaluate.py
│   ├── export_reports.py
│   └── generate_reports.py
├── data/
│   └── synthetic/
│       ├── raw/
│       └── gold/
├── schemas/
│   └── visit_schema_v1.json
└── tests/
    ├── test_normalize.py
    └── test_pipeline_rules.py