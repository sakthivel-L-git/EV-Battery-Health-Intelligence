import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="EV Battery Health Intelligence",
    page_icon="🔋",
    layout="wide",
)

PROJECT_PATH = Path(__file__).resolve().parent
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
        joblib.load(MODELS_PATH / "risk_random_forest.joblib"),
        joblib.load(MODELS_PATH / "anomaly_isolation_forest.joblib"),
    )


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero {
            padding: 1.5rem 1.8rem;
            border-radius: 18px;
            background: linear-gradient(120deg, #0b2742, #123d60);
            border: 1px solid #24587d;
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            margin: 0;
            color: #f2f7fb;
            font-size: 2.2rem;
        }

        .hero p {
            margin: 0.5rem 0 0;
            color: #bfd4e5;
        }

        [data-testid="stMetric"] {
            background: #172433;
            border: 1px solid #29465e;
            border-radius: 14px;
            padding: 1rem;
        }

        [data-testid="stMetricLabel"] {
            color: #a8c9df;
        }

        .status-box {
            padding: 1rem 1.3rem;
            border-radius: 12px;
            font-size: 1.05rem;
            font-weight: 600;
            margin: 0.8rem 0 1.5rem;
        }

        .healthy {
            background: #123c2d;
            color: #b8f5d2;
            border: 1px solid #218354;
        }

        .monitor {
            background: #4a3810;
            color: #ffe4a0;
            border: 1px solid #9b7522;
        }

        .maintenance {
            background: #4a2025;
            color: #ffc3c7;
            border: 1px solid #9b3a44;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

data = load_data()
soh_package, rul_package, risk_package, anomaly_package = load_models()

soh_model = soh_package["model"]
rul_model = rul_package["model"]
risk_model = risk_package["model"]

anomaly_model = anomaly_package["model"]
anomaly_scaler = anomaly_package["scaler"]
anomaly_features = anomaly_package["feature_columns"]

feature_columns = soh_package["feature_columns"]


def forest_prediction_interval(model, features):
    """Return the ensemble prediction and a simple 10%-90% uncertainty band."""
    tree_predictions = np.array(
        [tree.predict(features)[0] for tree in model.estimators_]
    )
    return (
        float(tree_predictions.mean()),
        float(np.quantile(tree_predictions, 0.10)),
        float(np.quantile(tree_predictions, 0.90)),
    )


def health_summary(soh, rul, anomaly):
    rul_component = np.clip(rul, 0, 100)
    anomaly_component = 100 if anomaly == "Normal" else 0
    score = 0.55 * soh + 0.35 * rul_component + 0.10 * anomaly_component

    if score >= 80:
        return float(score), "Healthy", "healthy"
    if score >= 60:
        return float(score), "Monitor", "monitor"
    return float(score), "Maintenance Recommended", "maintenance"

st.markdown(
    """
    <div class="hero">
        <h1>🔋 EV Battery Health Intelligence</h1>
        <p>Predictive maintenance dashboard powered by State of Health,
        Remaining Useful Life, anomaly detection, and risk classification.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Control Panel")
st.sidebar.caption("EV Battery Health Intelligence | Capstone Edition")
st.sidebar.caption("Choose a battery and cycle to analyse.")

battery_id = st.sidebar.selectbox(
    "Battery ID",
    sorted(data["battery_id"].unique()),
)

battery_history = data[
    data["battery_id"] == battery_id
].sort_values("discharge_cycle").copy()

selected_cycle = st.sidebar.slider(
    "Discharge Cycle",
    min_value=int(battery_history["discharge_cycle"].min()),
    max_value=int(battery_history["discharge_cycle"].max()),
    value=int(battery_history["discharge_cycle"].max()),
)

st.sidebar.divider()
st.sidebar.subheader("Digital Twin Scenario")
st.sidebar.caption("Test a modified operating condition before using it.")
temperature_shift = st.sidebar.slider("Temperature change (°C)", -5, 15, 0)
voltage_shift = st.sidebar.slider("Mean-voltage change (V)", -0.15, 0.15, 0.00, 0.01)
duration_multiplier = st.sidebar.slider("Discharge-duration multiplier", 0.70, 1.40, 1.00, 0.05)
cycles_per_day = st.sidebar.slider("Expected cycles per day", 1, 6, 2)

battery_history["capacity_change_ah"] = (
    battery_history["capacity_ah"].diff().fillna(0)
)

selected_row = battery_history[
    battery_history["discharge_cycle"] == selected_cycle
].copy()

predicted_soh, soh_lower, soh_upper = forest_prediction_interval(
    soh_model, selected_row[feature_columns]
)

predicted_rul, rul_lower, rul_upper = forest_prediction_interval(
    rul_model, selected_row[feature_columns]
)
predicted_rul = max(0, predicted_rul)
rul_lower = max(0, rul_lower)
rul_upper = max(0, rul_upper)

predicted_risk = risk_model.predict(
    selected_row[feature_columns]
)[0]

anomaly_input = anomaly_scaler.transform(
    selected_row[anomaly_features]
)

anomaly_prediction = anomaly_model.predict(anomaly_input)[0]
anomaly_status = "Anomaly" if anomaly_prediction == -1 else "Normal"
anomaly_score = float(-anomaly_model.score_samples(anomaly_input)[0])

health_score, maintenance_status, status_class = health_summary(
    predicted_soh, predicted_rul, anomaly_status
)

# Digital Twin: simulate a changed operating profile using the same trained models.
scenario_row = selected_row.copy()
scenario_row["mean_temperature_c"] += temperature_shift
scenario_row["max_temperature_c"] += temperature_shift
scenario_row["mean_voltage_v"] += voltage_shift
scenario_row["discharge_duration_s"] *= duration_multiplier

scenario_soh, scenario_soh_lower, scenario_soh_upper = forest_prediction_interval(
    soh_model, scenario_row[feature_columns]
)
scenario_rul, scenario_rul_lower, scenario_rul_upper = forest_prediction_interval(
    rul_model, scenario_row[feature_columns]
)
scenario_rul = max(0, scenario_rul)
scenario_risk = risk_model.predict(scenario_row[feature_columns])[0]

scenario_anomaly_input = anomaly_scaler.transform(
    scenario_row[anomaly_features]
)
scenario_anomaly_prediction = anomaly_model.predict(scenario_anomaly_input)[0]
scenario_anomaly_status = (
    "Anomaly" if scenario_anomaly_prediction == -1 else "Normal"
)
scenario_health_score, scenario_maintenance_status, _ = health_summary(
    scenario_soh, scenario_rul, scenario_anomaly_status
)

scenario_changed = (
    temperature_shift != 0
    or voltage_shift != 0
    or duration_multiplier != 1.0
)

st.subheader(f"Battery {battery_id} · Cycle {selected_cycle}")

metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)

metric_1.metric("Predicted SoH", f"{predicted_soh:.1f}%")
metric_2.metric("Predicted RUL", f"{predicted_rul:.1f} cycles")
metric_3.metric("Risk Level", predicted_risk)
metric_4.metric("Anomaly Status", anomaly_status)
metric_5.metric("Health Score", f"{health_score:.1f}/100")

st.markdown(
    f"""
    <div class="status-box {status_class}">
        Recommendation: {maintenance_status}
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"Model uncertainty (10th-90th percentile across Random Forest trees): "
    f"SoH {soh_lower:.1f}% to {soh_upper:.1f}% | "
    f"RUL {rul_lower:.1f} to {rul_upper:.1f} cycles."
)

chart_left, chart_right = st.columns([1.2, 1])

with chart_left:
    st.subheader("Battery Health Trend")

    soh_figure = go.Figure()

    soh_figure.add_trace(
        go.Scatter(
            x=battery_history["discharge_cycle"],
            y=battery_history["soh_percent"],
            mode="lines+markers",
            name="Measured SoH",
            line=dict(color="#42a5f5", width=3),
            marker=dict(size=4),
        )
    )

    soh_figure.add_trace(
        go.Scatter(
            x=[selected_cycle],
            y=[predicted_soh],
            mode="markers",
            name="Predicted SoH",
            marker=dict(color="#ffca28", size=13, symbol="diamond"),
        )
    )

    soh_figure.add_hline(
        y=80,
        line_dash="dash",
        line_color="#ef5350",
        annotation_text="80% maintenance threshold",
        annotation_position="bottom right",
    )

    soh_figure.update_layout(
        template="plotly_dark",
        height=365,
        margin=dict(l=15, r=15, t=40, b=15),
        paper_bgcolor="#10151e",
        plot_bgcolor="#10151e",
        xaxis_title="Discharge Cycle",
        yaxis_title="State of Health (%)",
        yaxis_range=[65, 105],
        legend=dict(orientation="h", y=1.12),
    )

    st.plotly_chart(
        soh_figure,
        use_container_width=True,
        config={"displayModeBar": False},
    )

with chart_right:
    st.subheader("Health Score Gauge")

    gauge_figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=health_score,
            number={"suffix": " / 100", "font": {"size": 40}},
            title={"text": "Overall Battery Health"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#42a5f5"},
                "steps": [
                    {"range": [0, 60], "color": "#5d252b"},
                    {"range": [60, 80], "color": "#5b471d"},
                    {"range": [80, 100], "color": "#1c4c38"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 4},
                    "thickness": 0.75,
                    "value": health_score,
                },
            },
        )
    )

    gauge_figure.update_layout(
        template="plotly_dark",
        height=365,
        margin=dict(l=20, r=20, t=70, b=15),
        paper_bgcolor="#10151e",
    )

    st.plotly_chart(
        gauge_figure,
        use_container_width=True,
        config={"displayModeBar": False},
    )

st.divider()

trend_left, trend_right = st.columns(2)

with trend_left:
    st.subheader("Capacity Degradation and Anomalies")

    history_anomaly_input = anomaly_scaler.transform(
        battery_history[anomaly_features]
    )

    battery_history["anomaly_prediction"] = anomaly_model.predict(
        history_anomaly_input
    )

    anomalies = battery_history[
        battery_history["anomaly_prediction"] == -1
    ]

    capacity_figure = go.Figure()

    capacity_figure.add_trace(
        go.Scatter(
            x=battery_history["discharge_cycle"],
            y=battery_history["capacity_ah"],
            mode="lines",
            name="Capacity",
            line=dict(color="#80cbc4", width=3),
        )
    )

    capacity_figure.add_trace(
        go.Scatter(
            x=anomalies["discharge_cycle"],
            y=anomalies["capacity_ah"],
            mode="markers",
            name="Detected anomaly",
            marker=dict(color="#ff5252", size=10, symbol="x"),
        )
    )

    capacity_figure.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(l=15, r=15, t=40, b=15),
        paper_bgcolor="#10151e",
        plot_bgcolor="#10151e",
        xaxis_title="Discharge Cycle",
        yaxis_title="Capacity (Ah)",
        legend=dict(orientation="h", y=1.12),
    )

    st.plotly_chart(
        capacity_figure,
        use_container_width=True,
        config={"displayModeBar": False},
    )

with trend_right:
    st.subheader("Decision Explanation")

    reasons = []

    if predicted_soh < 80:
        reasons.append("State of Health is below the 80% maintenance threshold.")
    elif predicted_soh < 90:
        reasons.append("State of Health is declining and should be monitored.")
    else:
        reasons.append("State of Health remains in the healthy operating range.")

    if predicted_rul < 10:
        reasons.append("Remaining Useful Life is critically low.")
    elif predicted_rul < 30:
        reasons.append("Remaining Useful Life is limited; plan maintenance soon.")
    else:
        reasons.append("Remaining Useful Life is currently acceptable.")

    if anomaly_status == "Anomaly":
        reasons.append("The sensor pattern is unusual compared with normal cycles.")
    else:
        reasons.append("No unusual sensor pattern was detected in this cycle.")

    for reason in reasons:
        st.info(reason)

    st.caption(
        f"Anomaly score: {anomaly_score:.3f} "
        "(higher values indicate more unusual behaviour)."
    )

st.divider()

st.subheader("Digital Twin Scenario Lab")
st.caption(
    "Adjust the operating controls in the sidebar to compare the current "
    "battery state with a simulated operating condition."
)

scenario_left, scenario_middle, scenario_right = st.columns(3)

with scenario_left:
    st.metric(
        "Scenario Health Score",
        f"{scenario_health_score:.1f}/100",
        delta=f"{scenario_health_score - health_score:+.1f}",
        delta_color="normal",
    )
    st.metric(
        "Scenario SoH",
        f"{scenario_soh:.1f}%",
        delta=f"{scenario_soh - predicted_soh:+.1f}%",
        delta_color="normal",
    )

with scenario_middle:
    st.metric(
        "Scenario RUL",
        f"{scenario_rul:.1f} cycles",
        delta=f"{scenario_rul - predicted_rul:+.1f}",
        delta_color="normal",
    )
    st.metric("Scenario Risk", scenario_risk)

with scenario_right:
    estimated_days = predicted_rul / cycles_per_day
    scenario_days = scenario_rul / cycles_per_day
    service_date = pd.Timestamp.today() + pd.Timedelta(days=estimated_days)

    st.metric("Baseline Service Window", f"{estimated_days:.0f} days")
    st.metric("Scenario Service Window", f"{scenario_days:.0f} days")
    st.caption(
        f"Suggested inspection date: {service_date.strftime('%d %b %Y')} "
        f"at {cycles_per_day} expected cycles/day."
    )

if scenario_changed:
    if scenario_health_score < health_score:
        st.warning(
            f"The simulated profile changes the recommendation to: "
            f"{scenario_maintenance_status}."
        )
    else:
        st.success(
            f"The simulated profile maintains or improves the health score. "
            f"Recommendation: {scenario_maintenance_status}."
        )
else:
    st.info(
        "Try a warmer temperature, lower voltage, or longer discharge in the "
        "sidebar to run a what-if digital-twin scenario."
    )

with st.expander("Why this makes the system different"):
    st.write(
        "This is not only a historical dashboard. The Digital Twin Scenario "
        "Lab uses the trained models to estimate how a changed operating "
        "profile may affect battery health, remaining life, anomaly status, "
        "risk, and maintenance timing."
    )

st.divider()

st.subheader("Battery Passport")
st.caption(
    "A concise, exportable health record for the selected battery and cycle."
)

passport_left, passport_middle, passport_right = st.columns(3)

with passport_left:
    st.metric("Battery Identity", battery_id)
    st.metric("Current Cycle", selected_cycle)

with passport_middle:
    st.metric("Operating Temperature", f"{selected_row['max_temperature_c'].iloc[0]:.1f} °C")
    st.metric("Measured Capacity", f"{selected_row['capacity_ah'].iloc[0]:.3f} Ah")

with passport_right:
    st.metric("Maintenance Status", maintenance_status)
    st.metric("Suggested Service Date", service_date.strftime("%d %b %Y"))

passport = {
    "project_title": "EV Battery Health Intelligence",
    "battery_id": battery_id,
    "discharge_cycle": int(selected_cycle),
    "predicted_soh_percent": round(predicted_soh, 2),
    "predicted_rul_cycles": round(predicted_rul, 2),
    "risk_level": predicted_risk,
    "anomaly_status": anomaly_status,
    "health_score": round(health_score, 2),
    "maintenance_recommendation": maintenance_status,
    "suggested_service_date": service_date.strftime("%Y-%m-%d"),
    "model_uncertainty": {
        "soh_10th_to_90th_percentile": [round(soh_lower, 2), round(soh_upper, 2)],
        "rul_10th_to_90th_percentile": [round(rul_lower, 2), round(rul_upper, 2)],
    },
}

st.download_button(
    "Download Battery Passport (JSON)",
    data=json.dumps(passport, indent=2),
    file_name=f"{battery_id}_cycle_{selected_cycle}_battery_passport.json",
    mime="application/json",
)

with st.expander("Capstone Innovation Summary"):
    st.markdown(
        """
        **EV Battery Health Intelligence** integrates four ML tasks into one
        predictive-maintenance workflow: State of Health prediction, Remaining
        Useful Life prediction, anomaly detection, and risk classification.

        The project goes beyond basic prediction by adding model uncertainty,
        fleet-level analytics, a Battery Passport export, and a Digital Twin
        Scenario Lab for testing operating-stress what-if cases.
        """
    )

st.subheader("Selected Cycle Sensor Data")

sensor_data = selected_row[
    [
        "mean_temperature_c",
        "max_temperature_c",
        "mean_voltage_v",
        "discharge_duration_s",
        "capacity_ah",
        "capacity_change_ah",
    ]
].T.reset_index()

sensor_data.columns = ["Sensor Feature", "Value"]

sensor_data["Sensor Feature"] = sensor_data["Sensor Feature"].replace(
    {
        "mean_temperature_c": "Mean Temperature (°C)",
        "max_temperature_c": "Maximum Temperature (°C)",
        "mean_voltage_v": "Mean Voltage (V)",
        "discharge_duration_s": "Discharge Duration (s)",
        "capacity_ah": "Measured Capacity (Ah)",
        "capacity_change_ah": "Capacity Change (Ah)",
    }
)

st.dataframe(
    sensor_data,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "Educational predictive-maintenance decision-support system. "
    "It is not validated for real vehicle safety decisions."
)
