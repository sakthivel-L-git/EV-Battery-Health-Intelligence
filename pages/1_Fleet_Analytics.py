from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Fleet Analytics", page_icon="📊", layout="wide")

PROJECT_PATH = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_PATH / "data" / "processed" / "battery_discharge_cycles.csv"
MODELS_PATH = PROJECT_PATH / "models"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_models():
    soh = joblib.load(MODELS_PATH / "soh_random_forest.joblib")
    rul = joblib.load(MODELS_PATH / "rul_random_forest.joblib")
    risk = joblib.load(MODELS_PATH / "risk_random_forest.joblib")
    return soh, rul, risk


data = load_data()
soh_package, rul_package, risk_package = load_models()
feature_columns = soh_package["feature_columns"]

latest = (
    data.sort_values("discharge_cycle")
    .groupby("battery_id", as_index=False)
    .tail(1)
    .copy()
)

latest["predicted_soh"] = soh_package["model"].predict(latest[feature_columns])
latest["predicted_rul"] = np.maximum(
    rul_package["model"].predict(latest[feature_columns]), 0
)
latest["predicted_risk"] = risk_package["model"].predict(latest[feature_columns])

initial_capacity = data.groupby("battery_id")["capacity_ah"].first()
latest["initial_capacity_ah"] = latest["battery_id"].map(initial_capacity)
latest["capacity_loss_percent"] = (
    1 - latest["capacity_ah"] / latest["initial_capacity_ah"]
) * 100

# Some NASA cycles can show a small measured capacity recovery. Plotly bubble
# sizes must never be negative, so use a non-negative visual severity value.
latest["capacity_loss_percent"] = latest["capacity_loss_percent"].clip(lower=0)
latest["bubble_size"] = latest["capacity_loss_percent"].clip(lower=0.2)

st.title("📊 Fleet Analytics Center")
st.caption("Compare every monitored battery as a fleet, not as isolated cycles.")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Batteries Monitored", len(latest))
metric_2.metric("Fleet Avg. SoH", f"{latest['predicted_soh'].mean():.1f}%")
metric_3.metric("High-Risk Batteries", int((latest["predicted_risk"] == "High").sum()))
metric_4.metric("Fleet Avg. RUL", f"{latest['predicted_rul'].mean():.1f} cycles")

overview_left, overview_right = st.columns(2)

with overview_left:
    st.subheader("Fleet Health Ranking")
    health_rank = latest.sort_values("predicted_soh")
    figure = px.bar(
        health_rank,
        x="predicted_soh",
        y="battery_id",
        orientation="h",
        color="predicted_risk",
        color_discrete_map={"Low": "#43a047", "Medium": "#fbc02d", "High": "#e53935"},
        text="predicted_soh",
        labels={"predicted_soh": "Predicted SoH (%)", "battery_id": "Battery"},
    )
    figure.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    figure.update_layout(
        template="plotly_dark",
        height=400,
        paper_bgcolor="#10151e",
        plot_bgcolor="#10151e",
        margin=dict(l=10, r=15, t=20, b=10),
        xaxis_range=[0, 110],
        legend_title_text="Risk",
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

with overview_right:
    st.subheader("SoH vs Remaining Useful Life")
    figure = px.scatter(
        latest,
        x="predicted_soh",
        y="predicted_rul",
        size="bubble_size",
        color="predicted_risk",
        hover_name="battery_id",
        color_discrete_map={"Low": "#43a047", "Medium": "#fbc02d", "High": "#e53935"},
        labels={
            "predicted_soh": "Predicted SoH (%)",
            "predicted_rul": "Predicted RUL (cycles)",
            "bubble_size": "Capacity loss (%)",
        },
        size_max=45,
    )
    figure.update_layout(
        template="plotly_dark",
        height=400,
        paper_bgcolor="#10151e",
        plot_bgcolor="#10151e",
        margin=dict(l=10, r=15, t=20, b=10),
        legend_title_text="Risk",
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

st.subheader("Maintenance Priority Board")
priority = latest[
    [
        "battery_id",
        "discharge_cycle",
        "predicted_soh",
        "predicted_rul",
        "predicted_risk",
        "capacity_loss_percent",
        "max_temperature_c",
    ]
].sort_values(["predicted_risk", "predicted_rul"], ascending=[True, True])

priority = priority.rename(
    columns={
        "battery_id": "Battery",
        "discharge_cycle": "Latest Cycle",
        "predicted_soh": "Predicted SoH (%)",
        "predicted_rul": "Predicted RUL (cycles)",
        "predicted_risk": "Risk Level",
        "capacity_loss_percent": "Capacity Loss (%)",
        "max_temperature_c": "Max Temperature (°C)",
    }
)

st.dataframe(
    priority.style.format(
        {
            "Predicted SoH (%)": "{:.1f}",
            "Predicted RUL (cycles)": "{:.1f}",
            "Capacity Loss (%)": "{:.1f}",
            "Max Temperature (°C)": "{:.1f}",
        }
    ),
    hide_index=True,
    use_container_width=True,
)

st.download_button(
    "Download Fleet Maintenance Board (CSV)",
    data=priority.to_csv(index=False).encode("utf-8"),
    file_name="fleet_maintenance_board.csv",
    mime="text/csv",
)

st.caption(
    "The fleet view is a decision-support prioritization tool. It is based on "
    "the educational NASA battery-aging dataset and is not a vehicle-safety system."
)
