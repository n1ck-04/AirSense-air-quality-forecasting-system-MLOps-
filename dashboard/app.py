"""AirSense dashboard (Streamlit).

Reads only the files in reports/, which train.py and monitor.py produce, so it
can be deployed on its own (e.g. Streamlit Community Cloud) without the raw
data or the model. If the model is available locally, a live-forecast panel
appears too.

Run:  streamlit run dashboard/app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CATEGORY_COLOURS = {"Good": "#55a84f", "Satisfactory": "#a3c853", "Moderate": "#fff833",
                    "Poor": "#f29c33", "Very Poor": "#e93f33", "Severe": "#af2d24"}

st.set_page_config(page_title="AirSense", page_icon="🌫️", layout="wide")


@st.cache_data
def load():
    fc = pd.read_parquet(REPORTS / "test_forecasts.parquet")
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    monitoring = json.loads((REPORTS / "monitoring.json").read_text())
    weekly = pd.read_csv(REPORTS / "monitoring_weekly.csv", parse_dates=["week"])
    return fc, metrics, monitoring, weekly


fc, metrics, monitoring, weekly = load()

st.title("AirSense: 3-day PM2.5 forecasts for Indian cities")
st.caption("Test year 1 Jul 2019 – 1 Jul 2020, 17 cities, CPCB data. The model never saw this year during training. "
           "Categories use the CPCB PM2.5 sub-index only.")

tab_fc, tab_quality, tab_monitor, tab_live = st.tabs(["Forecasts", "How accurate?", "Monitoring", "Live forecast"])

# --------------------------------------------------------------------------- forecasts
with tab_fc:
    c1, c2 = st.columns([1, 1])
    city = c1.selectbox("City", sorted(fc.city.unique()), index=sorted(fc.city.unique()).index("Delhi"))
    horizon = c2.radio("Forecast made", [1, 2, 3], horizontal=True, format_func=lambda h: f"{h} day{'s' if h > 1 else ''} ahead")
    d = fc[(fc.city == city) & (fc.horizon == horizon)].sort_values("target_date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.target_date, y=d.hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d.target_date, y=d.lo, fill="tonexty", line=dict(width=0), name="80% interval",
                             fillcolor="rgba(31,119,180,0.2)"))
    fig.add_trace(go.Scatter(x=d.target_date, y=d.actual, name="Actual", line=dict(color="black", width=1.3)))
    fig.add_trace(go.Scatter(x=d.target_date, y=d.forecast, name="AirSense forecast", line=dict(width=1.3)))
    fig.add_trace(go.Scatter(x=d.target_date, y=d.persistence, name="Persistence (today's value)",
                             line=dict(width=1, dash="dot"), visible="legendonly"))
    fig.add_vline(x=pd.Timestamp("2020-03-25").timestamp() * 1000, line_dash="dash", line_color="grey")
    fig.add_hline(y=90, line_dash="dot", line_color="#f29c33", annotation_text="Poor (90)")
    fig.update_layout(height=430, margin=dict(t=20, b=20), yaxis_title="PM2.5 (µg/m³)", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    ok = d.dropna(subset=["persistence"])
    m_mae, p_mae = (ok.forecast - ok.actual).abs().mean(), (ok.persistence - ok.actual).abs().mean()
    a, b, c = st.columns(3)
    a.metric("AirSense error (MAE)", f"{m_mae:.1f} µg/m³")
    b.metric("Persistence error", f"{p_mae:.1f} µg/m³")
    c.metric("Error removed vs persistence", f"{100 * (1 - m_mae / p_mae):.0f}%")

# ----------------------------------------------------------------------------- quality
with tab_quality:
    rows = []
    for h in (1, 2, 3):
        m = metrics[f"h{h}"]
        rows.append({"Horizon": f"{h} day{'s' if h > 1 else ''}", "AirSense MAE": m["model"]["mae"],
                     "Persistence MAE": m["persistence"]["mae"], "7-day mean MAE": m["mean7"]["mae"],
                     "Skill vs persistence": f"{100 * m['skill_vs_persistence']:.0f}%",
                     "Skill vs best baseline": f"{100 * m['skill_vs_best_baseline']:.0f}%",
                     "Category accuracy": f"{100 * m['model']['category_accuracy']:.0f}%",
                     "'Poor' alert precision": f"{100 * m['model']['poor_alert_precision']:.0f}%",
                     "'Poor' alert recall": f"{100 * m['model']['poor_alert_recall']:.0f}%",
                     "80% interval coverage": f"{100 * m['model']['interval_coverage']:.0f}%"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.image(str(REPORTS / "error_by_horizon.png"), width=620)
    st.subheader("By city (1 day ahead)")
    st.dataframe(pd.DataFrame(metrics["per_city_h1"]), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- monitoring
with tab_monitor:
    mon = monitoring["monitoring"]
    st.markdown(
        f"**Naive drift check** (PSI > 0.2) raised an alarm in **{mon['naive_psi_alarm_weeks']} of {mon['weeks']}** weeks. "
        f"It mostly reacts to the monsoon and winter, which happen every year. "
        f"The **season-adjusted check** raised **{mon['seasonal_alarm_weeks']}** and first fired on "
        f"**{mon['first_seasonal_alarm_after_lockdown']}**, days into the lockdown. The model's skill stayed above zero "
        f"in {mon['weeks'] - mon['skill_alarm_weeks']} of {mon['weeks']} weeks.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly.week, y=weekly.naive_psi, name="Naive PSI"))
    fig.add_trace(go.Scatter(x=weekly.week, y=weekly.seasonal_shift, name="Season-adjusted shift (z)"))
    fig.add_trace(go.Scatter(x=weekly.week, y=weekly.skill_28d * 10, name="Skill vs persistence ×10"))
    fig.add_vrect(x0=pd.Timestamp("2020-03-25"), x1=weekly.week.max(), fillcolor="grey", opacity=0.15, line_width=0,
                  annotation_text="lockdown")
    fig.update_layout(height=420, margin=dict(t=20, b=20), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    pol = monitoring["policies"]
    st.dataframe(pd.DataFrame([{"Policy": k, "Year MAE": v["year"]["mae"], "Lockdown MAE": v["lockdown"]["mae"],
                                "Retrains": v["retrains"]} for k, v in pol.items()]), hide_index=True)

# --------------------------------------------------------------------------------- live
with tab_live:
    try:
        from airsense.forecast import Forecaster

        @st.cache_resource
        def forecaster():
            return Forecaster.from_disk()

        f = forecaster()
        c1, c2 = st.columns(2)
        lcity = c1.selectbox("City ", f.cities, index=f.cities.index("Delhi") if "Delhi" in f.cities else 0)
        as_of = c2.date_input("Forecast as of", value=pd.Timestamp("2019-11-01"),
                              min_value=f.first_day, max_value=f.last_day)
        r = f.forecast(lcity, pd.Timestamp(as_of))
        st.write(f"Measured on {r['as_of']}: **{r['observed_pm25']} µg/m³**" if r["observed_pm25"] else
                 f"No measurement on {r['as_of']}.")
        cols = st.columns(3)
        for col, day in zip(cols, r["forecasts"], strict=True):
            colour = CATEGORY_COLOURS.get(day["category"], "#ccc")
            col.markdown(f"<div style='border-left:8px solid {colour};padding:8px 12px'>"
                         f"<b>{day['date']}</b><br><span style='font-size:1.6em'>{day['pm25']:.0f}</span> µg/m³<br>"
                         f"{day['category']}<br><small>80% range {day['lo']:.0f}–{day['hi']:.0f}</small></div>",
                         unsafe_allow_html=True)
    except Exception as e:  # noqa: BLE001  (no local model, e.g. on Streamlit Cloud)
        st.info("The live forecast needs the trained model on this machine (run `make all`). "
                f"The other tabs work without it. ({type(e).__name__})")
