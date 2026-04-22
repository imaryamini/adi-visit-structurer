# ADI Assistant
> Convert clinical home-care notes into structured ADI reports in seconds.

ADI Assistant is a prototype system developed during an internship focused on **ADI (Assistenza Domiciliare Integrata)** clinical documentation.

The goal is to support healthcare professionals by turning **free-text or dictated home-care visit notes** into **structured clinical report drafts**, aligned with real ADI workflows.

The system is designed to reduce manual documentation effort while improving **consistency and usability of clinical data**.

---

## Key Features

* 🎤 Voice-to-text input for clinical dictation  
* 📝 Manual text input for flexibility  
* 🧠 Hybrid extraction pipeline (rule-based + LLM)  
* 📊 Structured ADI-style outputs  
* ⚠️ Quality checks for missing or inconsistent data  
* 🌐 Web dashboard for interactive use  

---

## What the System Extracts

From a single clinical note, the assistant generates:

* Reason for visit  
* Vital signs  
  * Blood pressure  
  * Heart rate  
  * Temperature  
  * SpO₂  
* Interventions performed  
* Follow-up indications  
* Critical issues (if present)  
* Quality warnings  

---

## System Architecture

The system follows a modular pipeline:

1. **Input**
   * Voice recording or typed clinical note  

2. **Preprocessing**
   * Text cleaning and normalization  

3. **Extraction**
   * Rule-based methods (for structured data such as vitals)  
   * LLM-based extraction (via Ollama, e.g. LLaMA 3.1)  

4. **Post-processing**
   * Label normalization  
   * Deduplication and formatting  

5. **Quality Checks**
   * Missing mandatory fields  
   * Logical inconsistencies  

6. **Output**
   * Structured JSON (ADI-compatible)  
   * Optional dashboard view  

---

## Hybrid Approach

The system combines:

* **Rule-based methods**
  * High precision for numeric and structured data  
  * Deterministic and interpretable  

* **LLM-based extraction**
  * Handles variability in clinical language  
  * Improves recall in less structured notes  

This approach balances **precision and flexibility**, making the system more robust for real-world inputs.

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