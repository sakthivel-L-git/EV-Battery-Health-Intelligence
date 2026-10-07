# EV Battery Health Intelligence

EV Battery Health Intelligence is a multi-page machine-learning decision-support system for lithium-ion battery predictive maintenance. It uses the NASA Battery Data Set to estimate battery health, remaining useful life, anomalous behavior, and maintenance risk.

## Key capabilities

- State of Health (SoH) prediction with Random Forest regression
- Remaining Useful Life (RUL) prediction in discharge cycles
- Unsupervised anomaly detection with Isolation Forest
- Low, Medium, and High maintenance-risk classification
- Random Forest uncertainty intervals for SoH and RUL
- Digital Twin Scenario Lab for what-if operating-stress simulations
- Fleet Analytics, Model Insights, and Data Explorer pages
- Downloadable Battery Passport and maintenance-board exports

## Dataset

The project uses the NASA Prognostics Center of Excellence lithium-ion battery aging data. The processed dataset includes discharge-cycle capacity, voltage, temperature, duration, and derived SoH features for batteries B0005, B0006, B0007, and B0018.

## Project structure

```text
ev-battery-health/
├── app.py                         # Live digital-twin dashboard
├── pages/
│   ├── 1_Fleet_Analytics.py        # Fleet comparison and priority board
│   ├── 2_Model_Insights.py         # Metrics, explainability, and validation charts
│   └── 3_Data_Explorer.py          # Interactive dataset exploration
├── data/
│   ├── processed/                  # ML-ready CSV data
│   └── raw/                        # NASA source files
├── models/                         # Trained model artifacts
├── notebooks/                      # Training and exploration notebook
├── requirements.txt
└── README.md
```

## Run the application

```powershell
cd "$env:USERPROFILE\Documents\ev-battery-health"
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed in PowerShell, normally `http://localhost:8501`.

## Evaluation on an unseen battery

Battery B0018 is held out for evaluation.

| Task | Model | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| State of Health | Random Forest Regressor | 1.72 percentage points | 2.38 percentage points | 0.918 |
| Remaining Useful Life | Random Forest Regressor | 5.71 cycles | 8.35 cycles | 0.884 |
| Risk Classification | Random Forest Classifier | 82.6% accuracy | - | - |

## Project innovation

Most battery-health projects only predict SoH or RUL. This project integrates multiple ML tasks with a Digital Twin Scenario Lab. A user can adjust temperature, voltage, expected discharge duration, and cycles per day to compare baseline health with a simulated operating condition and maintenance timeline.

## Disclaimer

This is an educational predictive-maintenance decision-support project. It is not validated for real-world vehicle safety, warranty, or medical/safety-critical decisions.
