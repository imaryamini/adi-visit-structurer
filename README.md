# ADI Assistant  
**Transform clinical home-care notes into structured ADI report drafts through an automated pipeline.**

ADI Assistant is a prototype system developed during an internship, focused on **ADI (Assistenza Domiciliare Integrata)** clinical documentation.

The goal of the project is to support healthcare professionals by transforming **free-text or dictated home-care visit notes** into **structured clinical report drafts**, aligned with real ADI workflows.

The system is designed to reduce manual documentation effort while improving **consistency, readability, and usability of clinical data**.

---

## Key Features

- Voice-to-text input for clinical dictation  
- Manual text input for flexibility  
- Hybrid extraction pipeline (rule-based + LLM)  
- Structured ADI-style report generation  
- Quality checks for missing or inconsistent data  
- Web interface for interactive usage  

---

## What the System Extracts

From a single clinical note, the assistant generates:

- Reason for visit  
- Vital signs:
  - Blood pressure (systolic / diastolic)  
  - Heart rate  
  - Temperature  
  - SpO₂  
- Interventions performed  
- Follow-up indications  
- Critical issues (if present)  
- Quality warnings  

---

## System Architecture

The system follows a modular pipeline:

1. **Input**
   - Voice recording or typed clinical note  

2. **Preprocessing**
   - Text cleaning and normalization  

3. **Extraction**
   - Rule-based NLP (for structured data such as vital signs)  
   - LLM-based extraction (via Ollama, e.g. LLaMA 3.1)  

4. **Post-processing**
   - Label normalization (interventions, reasons, problems)  
   - Deduplication and formatting  

5. **Quality Checks**
   - Missing mandatory fields  
   - Logical inconsistencies  

6. **Output**
   - Structured JSON (ADI-compatible)  
   - Interactive dashboard view  

---

## Hybrid Approach

The system combines:

**Rule-based methods**
- High precision for numeric and structured data  
- Deterministic and interpretable  

**LLM-based extraction**
- Handles variability in clinical language  
- Improves recall and flexibility  

This hybrid design balances **reliability and adaptability**, making the system more robust in real-world scenarios.

---

## Example Output

```json
{
  "clinical": {
    "reason_for_visit": "controllo parametri",
    "vitals": {
      "blood_pressure_systolic": 130,
      "blood_pressure_diastolic": 80,
      "heart_rate": 72,
      "temperature": null,
      "spo2": 97
    },
    "interventions": ["monitoraggio_parametri_vitali"],
    "follow_up": "controllo tra 7 giorni"
  }
}
```

---

## Evaluation

The system includes an evaluation module comparing predictions with a reference dataset.

**Results (100 records):**

- Reason for visit accuracy: **0.74**  
- Follow-up accuracy: **0.44**  
- Vitals exact match rate: **0.70**  
- Interventions macro F1: **0.71**  
- Problems macro F1: **0.42**  

The system performs well on structured clinical data (vital signs and interventions) and achieves reasonable performance on semantic fields such as reason for visit and follow-up.

The main limitation remains **problem normalization**, due to the variability of clinical language and label granularity. This could be improved through ontology-based mapping or further LLM refinement.

---

## Project Structure

```
adi-visit-structurer/
├── app.py
├── src/
├── data/
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

- This is a prototype system developed for research and demonstration purposes  
- The dataset is synthetic and designed to resemble real ADI clinical notes  
- The system is not intended for clinical use  
- Focus: clarity, robustness, and practical workflow support  

---

## Future Improvements

- Integration with real clinical datasets  
- Improved speech recognition pipeline  
- Enhanced problem normalization (ontology-based mapping)  
- Improved LLM prompting and fine-tuning  
- User authentication and patient history tracking  
- Deployment as a full web application  

---

## Author

**Maryam Amini**  
Data Analysis Student  
University of Messina  

---

## Repository

https://github.com/imaryamini-code/adi-visit-structurer  

Developed as part of an internship in Data Analysis at the University of Messina.