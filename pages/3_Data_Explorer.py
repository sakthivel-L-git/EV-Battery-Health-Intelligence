from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Data Explorer", page_icon="🔎", layout="wide")

PROJECT_PATH = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_PATH / "data" / "processed" / "battery_discharge_cycles.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


data = load_data()

st.title("🔎 Battery Data Explorer")
st.caption(
    "Explore the processed NASA battery-aging records that feed the ML system."
)

st.sidebar.header("Explorer Filters")
selected_batteries = st.sidebar.multiselect(
    "Battery IDs",
    sorted(data["battery_id"].unique()),
    default=sorted(data["battery_id"].unique()),
)

maximum_cycle = int(data["discharge_cycle"].max())
cycle_range = st.sidebar.slider(
    "Discharge-cycle range",
    min_value=1,
    max_value=maximum_cycle,
    value=(1, maximum_cycle),
)

filtered = data[
    data["battery_id"].isin(selected_batteries)
    & data["discharge_cycle"].between(*cycle_range)
].copy()

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Visible Records", f"{len(filtered):,}")
metric_2.metric("Batteries Selected", len(selected_batteries))
metric_3.metric("Average Capacity", f"{filtered['capacity_ah'].mean():.3f} Ah")
metric_4.metric("Average Temperature", f"{filtered['mean_temperature_c'].mean():.1f} °C")

left, right = st.columns(2)

with left:
    st.subheader("Capacity Degradation by Battery")
    figure = px.line(
        filtered,
        x="discharge_cycle",
        y="capacity_ah",
        color="battery_id",
        markers=True,
        labels={"discharge_cycle": "Discharge Cycle", "capacity_ah": "Capacity (Ah)", "battery_id": "Battery"},
    )
    figure.update_layout(
        template="plotly_dark", height=390, paper_bgcolor="#10151e",
        plot_bgcolor="#10151e", margin=dict(l=10, r=15, t=15, b=10),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

with right:
    st.subheader("Temperature vs Capacity Scatter Plot")
    figure = px.scatter(
        filtered,
        x="mean_temperature_c",
        y="capacity_ah",
        color="battery_id",
        size="discharge_duration_s",
        hover_data=["discharge_cycle", "mean_voltage_v", "soh_percent"],
        labels={
            "mean_temperature_c": "Mean Temperature (°C)",
            "capacity_ah": "Capacity (Ah)",
            "battery_id": "Battery",
        },
        size_max=22,
    )
    figure.update_layout(
        template="plotly_dark", height=390, paper_bgcolor="#10151e",
        plot_bgcolor="#10151e", margin=dict(l=10, r=15, t=15, b=10),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

st.subheader("Feature Correlation Heatmap")
numeric_columns = [
    "discharge_cycle",
    "capacity_ah",
    "soh_percent",
    "mean_temperature_c",
    "max_temperature_c",
    "mean_voltage_v",
    "discharge_duration_s",
]
correlation = filtered[numeric_columns].corr().round(2)

figure = px.imshow(
    correlation,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
)
figure.update_layout(
    template="plotly_dark", height=500, paper_bgcolor="#10151e",
    plot_bgcolor="#10151e", margin=dict(l=10, r=15, t=20, b=10),
    coloraxis_colorbar_title="Correlation",
)
st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

st.subheader("Filtered Dataset")
display_columns = [
    "battery_id",
    "discharge_cycle",
    "capacity_ah",
    "soh_percent",
    "mean_temperature_c",
    "max_temperature_c",
    "mean_voltage_v",
    "discharge_duration_s",
]
st.dataframe(
    filtered[display_columns],
    hide_index=True,
    use_container_width=True,
    height=360,
)

st.download_button(
    "Download Filtered Data (CSV)",
    data=filtered[display_columns].to_csv(index=False).encode("utf-8"),
    file_name="filtered_battery_data.csv",
    mime="text/csv",
)
