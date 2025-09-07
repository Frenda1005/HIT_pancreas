# Pancreatic Cancer HIT Dosimetry Study

End-to-end research pipeline for Heavy Ion Therapy (HIT) dosimetry in pancreatic cancer:
- **Dosimetric characterization** (DVH metrics, overlays, radiomics)
- **Dose–response** (KM, Cox, RSF)
- **Normal-tissue toxicity** (ROC/Youden thresholds, NTCP logistic)
- **LKB NTCP** (EUD, TD50, m, probit MLE)
- **RBE vs physical dose** comparisons
- **Biomarker integration** (e.g., CA19-9)

> ⚠️ **PHI & DICOM**: do **not** commit patient data. This repo is code-only. All raw data stay in `data/` (gitignored).

---

## Quick start

### 1) Environment
```bash
# pip
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
