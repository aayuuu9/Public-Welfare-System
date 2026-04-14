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
