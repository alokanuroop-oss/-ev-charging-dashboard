
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

st.set_page_config(
    page_title="EV Charging Demand Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── load data ─────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_data
def load_data():
    y_orig           = np.load(f"{BASE}/y_orig.npy")
    pred_orig        = np.load(f"{BASE}/pred_orig.npy")
    per_station_rmse = np.load(f"{BASE}/per_station_rmse.npy")
    lat              = np.load(f"{BASE}/lat.npy")
    lon              = np.load(f"{BASE}/lon.npy")
    metrics_df       = pd.read_csv(f"{BASE}/metrics_per_horizon.csv")
    comp_df          = pd.read_csv(f"{BASE}/full_comparison.csv")
    return y_orig, pred_orig, per_station_rmse, lat, lon, metrics_df, comp_df

y_orig, pred_orig, per_station_rmse, lat, lon, metrics_df, comp_df = load_data()

N              = y_orig.shape[2]
horizon_labels = ["15min", "30min", "45min", "60min"]

# ── sidebar ───────────────────────────────────────────────────
st.sidebar.title("EV Charging Demand Forecast")
st.sidebar.markdown("**Model:** STGAT-GRU")
st.sidebar.markdown(f"**Stations:** {N}")
st.sidebar.markdown(f"**Test samples:** {y_orig.shape[0]}")
st.sidebar.markdown("---")

selected_station  = st.sidebar.slider("Station index", 0, N - 1, 0)
selected_horizon  = st.sidebar.selectbox("Forecast horizon", horizon_labels)
horizon_idx       = horizon_labels.index(selected_horizon)
n_show            = st.sidebar.slider("Timesteps to show", 50, 500, 200, step=50)

# ── tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Zone Forecast", "Prediction Error", "Spatial Heatmap", "Model Comparison"
])

# ── TAB 1: forecast vs actual ─────────────────────────────────
with tab1:
    st.subheader(f"Station {selected_station} — {selected_horizon} Forecast")

    actual = y_orig[-n_show:, horizon_idx, selected_station]
    pred   = pred_orig[-n_show:, horizon_idx, selected_station]
    steps  = list(range(len(actual)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=actual.tolist(), mode="lines",
                              name="Actual",
                              line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=steps, y=pred.tolist(), mode="lines",
                              name="Predicted",
                              line=dict(color="#ff7f0e", width=2, dash="dash")))
    fig.update_layout(xaxis_title="Test timestep",
                       yaxis_title="Occupancy", height=400,
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest Prediction — all horizons")
    cols = st.columns(4)
    for i, (h, col) in enumerate(zip(horizon_labels, cols)):
        a = float(y_orig[-1, i, selected_station])
        p = float(pred_orig[-1, i, selected_station])
        col.metric(label=f"+{h}", value=f"{p:.2f}",
                   delta=f"{p - a:+.2f} vs actual")

# ── TAB 2: error over time ────────────────────────────────────
with tab2:
    st.subheader(f"Prediction Error — Station {selected_station}, {selected_horizon}")

    err          = np.abs(y_orig[:, horizon_idx, selected_station]
                          - pred_orig[:, horizon_idx, selected_station])
    rolling_err  = pd.Series(err).rolling(24, min_periods=1).mean().values
    steps2       = list(range(len(err[-n_show:])))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=steps2, y=err[-n_show:].tolist(),
        mode="markers", name="Abs error",
        marker=dict(size=3, color="lightgray")
    ))
    fig2.add_trace(go.Scatter(
        x=steps2, y=rolling_err[-n_show:].tolist(),
        mode="lines", name="Rolling MAE (window=24)",
        line=dict(color="crimson", width=2)
    ))
    fig2.update_layout(xaxis_title="Timestep",
                        yaxis_title="Abs error", height=380)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Per-Station RMSE (15-min horizon)")
    fig3 = px.bar(
        x=list(range(N)), y=per_station_rmse.tolist(),
        labels={"x": "Station", "y": "RMSE"},
        title="RMSE by Station",
        color=per_station_rmse.tolist(),
        color_continuous_scale="RdYlGn_r"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── TAB 3: spatial heatmap ────────────────────────────────────
with tab3:
    st.subheader(f"Spatial Heatmap — {selected_horizon}")

    sample_idx  = st.slider("Test sample", 0, y_orig.shape[0] - 1,
                             y_orig.shape[0] - 1)
    actual_vals = y_orig[sample_idx, horizon_idx, :]
    pred_vals   = pred_orig[sample_idx, horizon_idx, :]
    err_vals    = np.abs(actual_vals - pred_vals)

    col1, col2 = st.columns(2)
    with col1:
        fig4 = px.scatter(
            x=lon.tolist(), y=lat.tolist(),
            color=actual_vals.tolist(),
            color_continuous_scale="YlOrRd",
            title="Actual Occupancy",
            labels={"x": "Lon", "y": "Lat", "color": "Occupancy"},
            size=(err_vals + 1).tolist()
        )
        fig4.update_layout(height=430)
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        fig5 = px.scatter(
            x=lon.tolist(), y=lat.tolist(),
            color=pred_vals.tolist(),
            color_continuous_scale="YlOrRd",
            title="Predicted Occupancy",
            labels={"x": "Lon", "y": "Lat", "color": "Occupancy"},
            size=(err_vals + 1).tolist()
        )
        fig5.update_layout(height=430)
        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Prediction Error Map")
    fig6 = px.scatter(
        x=lon.tolist(), y=lat.tolist(),
        color=err_vals.tolist(),
        color_continuous_scale="Reds",
        title="Absolute Prediction Error",
        labels={"x": "Lon", "y": "Lat", "color": "Abs Error"},
        size=(err_vals + 1).tolist()
    )
    fig6.update_layout(height=430)
    st.plotly_chart(fig6, use_container_width=True)

# ── TAB 4: model comparison ───────────────────────────────────
with tab4:
    st.subheader("Model Comparison — All Metrics")

    metric = st.radio("Metric", ["RMSE", "MAE"], horizontal=True)

    fig7 = px.bar(
        comp_df, x="Horizon", y=metric, color="Model",
        barmode="group",
        category_orders={"Horizon": horizon_labels},
        title=f"{metric} by Horizon and Model"
    )
    st.plotly_chart(fig7, use_container_width=True)

    st.subheader("Full Metrics Table")
    st.dataframe(
        comp_df[["Model", "Horizon", "MAE", "RMSE"]].round(4),
        use_container_width=True
    )
