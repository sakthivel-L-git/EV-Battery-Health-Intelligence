from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


st.set_page_config(page_title="Model Insights", page_icon="🧠", layout="wide")

PROJECT_PATH = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_PATH / "data" / "processed" / "battery_discharge_cycles.csv"
MODELS_PATH = PROJECT_PATH / "models"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_models():
    return (
        joblib.load(MODELS_PATH / "soh_random_forest.joblib"),
        joblib.load(MODELS_PATH / "rul_random_forest.joblib"),
    )


data = load_data()
soh_package, rul_package = load_models()
soh_model = soh_package["model"]
rul_model = rul_package["model"]
feature_columns = soh_package["feature_columns"]

held_out = data[data["battery_id"] == "B0018"].sort_values("discharge_cycle").copy()
held_out["predicted_soh"] = soh_model.predict(held_out[feature_columns])
held_out["predicted_rul"] = np.maximum(
    rul_model.predict(held_out[feature_columns]), 0
)

threshold_cycles = held_out.loc[
    held_out["soh_percent"] <= 80, "discharge_cycle"
]
eol_cycle = int(threshold_cycles.iloc[0]) if not threshold_cycles.empty else int(held_out["discharge_cycle"].max())
held_out["actual_rul"] = (eol_cycle - held_out["discharge_cycle"]).clip(lower=0)

soh_mae = mean_absolute_error(held_out["soh_percent"], held_out["predicted_soh"])
soh_rmse = root_mean_squared_error(held_out["soh_percent"], held_out["predicted_soh"])
soh_r2 = r2_score(held_out["soh_percent"], held_out["predicted_soh"])
rul_mae = mean_absolute_error(held_out["actual_rul"], held_out["predicted_rul"])
rul_rmse = root_mean_squared_error(held_out["actual_rul"], held_out["predicted_rul"])
rul_r2 = r2_score(held_out["actual_rul"], held_out["predicted_rul"])

st.title("🧠 Model Insights & Explainability")
st.caption("Evaluation uses B0018 as an unseen held-out battery.")

soh_col, rul_col = st.columns(2)

with soh_col:
    st.subheader("State of Health Model")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{soh_mae:.2f}%")
    c2.metric("RMSE", f"{soh_rmse:.2f}%")
    c3.metric("R²", f"{soh_r2:.3f}")

with rul_col:
    st.subheader("Remaining Useful Life Model")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{rul_mae:.2f} cycles")
    c2.metric("RMSE", f"{rul_rmse:.2f} cycles")
    c3.metric("R²", f"{rul_r2:.3f}")

left, right = st.columns(2)

with left:
    st.subheader("Feature Importance Bar Graph")
    importance = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance": soh_model.feature_importances_,
        }
    ).sort_values("Importance")

    figure = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        color="Importance",
        color_continuous_scale="Blues",
    )
    figure.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    figure.update_layout(
        template="plotly_dark",
        height=390,
        paper_bgcolor="#10151e",
        plot_bgcolor="#10151e",
        margin=dict(l=10, r=45, t=15, b=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

with right:
    st.subheader("Prediction Residual Bar Graph")
    residuals = held_out[["discharge_cycle", "soh_percent", "predicted_soh"]].copy()
    residuals["error"] = residuals["predicted_soh"] - residuals["soh_percent"]
    figure = px.bar(
        residuals,
        x="discharge_cycle",
        y="error",
        color="error",
        color_continuous_scale=["#ef5350", "#263238", "#42a5f5"],
        labels={"discharge_cycle": "Discharge Cycle", "error": "Prediction Error (percentage points)"},
    )
    figure.add_hline(y=0, line_color="#ffffff", line_width=1)
    figure.update_layout(
        template="plotly_dark",
        height=390,
        paper_bgcolor="#10151e",
        plot_bgcolor="#10151e",
        margin=dict(l=10, r=15, t=15, b=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

st.divider()

trend_left, trend_right = st.columns(2)

with trend_left:
    st.subheader("Actual vs Predicted SoH")
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=held_out["discharge_cycle"], y=held_out["soh_percent"],
        name="Actual SoH", line=dict(color="#42a5f5", width=3),
    ))
    figure.add_trace(go.Scatter(
        x=held_out["discharge_cycle"], y=held_out["predicted_soh"],
        name="Predicted SoH", line=dict(color="#ffca28", width=2, dash="dash"),
    ))
    figure.add_hline(y=80, line_dash="dot", line_color="#ef5350")
    figure.update_layout(
        template="plotly_dark", height=360, paper_bgcolor="#10151e",
        plot_bgcolor="#10151e", margin=dict(l=10, r=15, t=20, b=10),
        xaxis_title="Discharge Cycle", yaxis_title="SoH (%)",
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

with trend_right:
    st.subheader("Actual vs Predicted RUL")
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=held_out["discharge_cycle"], y=held_out["actual_rul"],
        name="Actual RUL", line=dict(color="#80cbc4", width=3),
    ))
    figure.add_trace(go.Scatter(
        x=held_out["discharge_cycle"], y=held_out["predicted_rul"],
        name="Predicted RUL", line=dict(color="#ffab91", width=2, dash="dash"),
    ))
    figure.update_layout(
        template="plotly_dark", height=360, paper_bgcolor="#10151e",
        plot_bgcolor="#10151e", margin=dict(l=10, r=15, t=20, b=10),
        xaxis_title="Discharge Cycle", yaxis_title="RUL (cycles)",
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

st.info(
    "Interpretation: feature importance measures how useful a feature was to "
    "this specific Random Forest model; it does not establish physical causality."
)
