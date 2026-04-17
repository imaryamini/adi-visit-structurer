# ADI Assistant
> Transform clinical home-care notes into structured ADI reports in seconds.

ADI Assistant is a prototype system developed within the context of an internship focused on **ADI (Assistenza Domiciliare Integrata)** clinical documentation.

The project aims to support healthcare professionals by transforming **free-text or dictated home-care visit notes** into **structured clinical report drafts**, aligned with real ADI workflows.

The system is designed to reduce manual documentation effort while improving **consistency, readability, and data usability**.

---

## Key Features

* 🎤 **Voice-to-text input** for clinical dictation
* 📝 **Manual text input** for flexibility
* 🧠 **Hybrid extraction pipeline** (rule-based + LLM)
* 📊 **Structured ADI-style outputs**
* ⚠️ **Quality checks** for missing or inconsistent data
* 🌐 **Web dashboard** for interactive usage

---

## What the System Extracts

From a single clinical note, the assistant generates:

* **Reason for visit**
* **Vital signs**

  * Blood pressure
  * Heart rate
  * Temperature
  * SpO₂
* **Interventions performed**
* **Follow-up indications**
* **Critical issues (if present)**
* **Quality warnings**

---

## System Architecture

The system follows a modular pipeline:

1. **Input**

   * Voice recording or typed clinical note

2. **Preprocessing**

   * Text cleaning and normalization

3. **Extraction**

   * Rule-based NLP (for structured data like vitals)
   * LLM-based extraction (via Ollama, e.g. LLaMA 3.1)

4. **Post-processing**

   * Label normalization (interventions, reasons)
   * Deduplication and formatting

5. **Quality Checks**

   * Missing mandatory fields
   * Logical inconsistencies

6. **Output**

   * Structured JSON (ADI-compatible)
   * Optional dashboard and reports

---

## Hybrid Approach

The system combines:

* **Rule-based methods**

  * High precision for numeric and structured data
  * Deterministic and interpretable

* **LLM-based extraction**

  * Handles variability in clinical language
  * Improves recall and flexibility

This hybrid design ensures a balance between **reliability and adaptability**, making the system more robust in real-world scenarios.

---

## Example Output

```json
{
  "clinical": {
    "reason_for_visit": "controllo parametri",
    "vitals": {
      "blood_pressure": "130/80",
      "heart_rate": "72",
      "temperature": null,
      "spo2": "97"
    },
    "interventions": ["monitoraggio_parametri_vitali"],
    "follow_up": "controllo tra 7 giorni"
  }
}
```

---

## Evaluation

The system includes an evaluation module comparing predictions with a gold dataset.

Metrics include:

* Reason for visit accuracy
* Follow-up accuracy
* Vitals exact match rate
* Interventions macro F1
* Problems macro F1

To run evaluation:

```bash
python3 src/evaluate.py
```

Results are saved in:

```
reports/metrics.json
```

---

## Project Structure

```
adi-visit-structurer/
├── app.py                  # Flask web app
├── src/
│   ├── preprocess.py
│   ├── extract_rules.py
│   ├── llm_extract.py
│   ├── normalize.py
│   ├── quality.py
│   ├── evaluate.py
│   └── run_pipeline.py
├── data/
│   └── synthetic/
├── reports/
├── templates/
├── static/
└── tests/
```

---

## How to Run

### 1. Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Start LLM (Ollama)

```bash
ollama serve
ollama pull llama3.1:8b
```

### 4. Run pipeline (batch mode)

```bash
python3 -m src.run_pipeline
```

### 5. Run web app

```bash
python3 app.py
```

### 6. Open dashboard

```
http://127.0.0.1:5000/assistant
```

---

## Notes

* This is a **prototype system**, developed for research and demonstration purposes
* The dataset is **synthetic**, designed to resemble real ADI notes
* The system is **not intended for clinical use**
* Focus: **clarity, robustness, and practical workflow support**

---

## Future Improvements

* Integration with real clinical datasets
* Better speech recognition pipeline
* Improved LLM prompting and fine-tuning
* User authentication and patient history tracking
* Deployment as a full web application

---

## Author

**Maryam Amini**
Data Analysis Student
University of Messina

---

## Repository

👉 Replace with your actual link:
https://github.com/imaryamini-code/adi-visit-structurer
