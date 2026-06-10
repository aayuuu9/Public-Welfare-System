# Fair ML for Public Welfare — Local Setup

## Files Included
- `ml_pipeline.py` — Full ML pipeline (dataset generation, training, fairness metrics, SHAP, DP)
- `results.json` — Pre-computed results from the pipeline
- `welfare_fairness_dashboard.html` — Interactive dashboard (open directly in any browser, no server needed)
- `welfare_fairness_report.docx` — Full research report (open in Word or Google Docs)

## Requirements
```
pip install numpy pandas scikit-learn matplotlib
```
Optional (for full SHAP values):
```
pip install shap
```
Optional (for AIF360 bias mitigation):
```
pip install aif360
```

## Run the Pipeline
```bash
python ml_pipeline.py
```
Outputs `results.json` with all metrics.

## View the Dashboard
Just open `welfare_fairness_dashboard.html` in Chrome, Firefox, or Safari.
No server required — it's fully self-contained.

## System Requirements
- Python 3.8+
- Node.js not required
- Works on Mac, Windows, Linux


This report presents a comprehensive end-to-end machine learning system designed to support fair, equitable, and auditable public welfare decisions. Leveraging synthetic datasets calibrated to SNAP, Medicaid, and Unemployment Insurance program characteristics, we demonstrate how algorithmic systems can replicate historical discrimination even when explicitly race-blind. We apply IBM AIF360 methodology to compute disparate impact, statistical parity difference, and equalized odds metrics; use permutation-based SHAP-style explanations to surface proxy discrimination; implement reweighing and threshold calibration for bias mitigation; and apply differential privacy (ε=2.0) to protect sensitive aggregate statistics. Key findings include a 23.9 percentage-point historical approval gap between White (47.8%) and Black (23.9%) applicants, a 2.9× false positive rate disparity in biased fraud detection that reduces to 1.8× after threshold equalization, and that income—rather than race—serves as the primary proxy variable perpetuating inequitable outcomes. We conclude with six policy recommendations for responsible deployment of welfare ML systems.
