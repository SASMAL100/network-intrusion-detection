import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="NetSentry | Network Intrusion Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Theme: "security operations console" — near-black canvas, monospace
# throughout (evokes a SOC terminal / packet log), cyan for clear traffic,
# crimson for flagged traffic, amber reserved for the sensitivity control.
# Deliberately distinct from a typical dashboard: no rounded SaaS cards,
# thin hairline borders instead, like a monitoring console panel.
# ----------------------------------------------------------------------------
SAFE = "#22D3EE"        # cyan — clear / normal traffic
ALERT = "#FF3B5C"       # crimson — flagged / intrusion
AMBER = "#FFB020"       # amber — sensitivity / warning zone
BG = "#0A0E14"
PANEL = "#0F1621"
LINE = "rgba(255,255,255,0.08)"
TEXT = "#DCE4EE"
MUTED = "#6B7686"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'JetBrains Mono', monospace;
}}
.stApp {{ background: {BG}; color: {TEXT}; }}
section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {LINE}; }}

.console-header {{
    padding: 1.4rem 1.8rem;
    background: {PANEL};
    border: 1px solid {LINE};
    border-left: 3px solid {SAFE};
    margin-bottom: 1.1rem;
}}
.console-header h1 {{
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: 0.01em;
}}
.console-header .accent {{ color: {SAFE}; }}
.console-header p {{
    font-family: 'Inter', sans-serif;
    color: {MUTED};
    margin-top: 0.4rem;
    font-size: 0.92rem;
}}
.blinking-dot {{
    display: inline-block;
    width: 8px; height: 8px;
    background: {SAFE};
    border-radius: 50%;
    margin-right: 8px;
    box-shadow: 0 0 8px {SAFE};
}}

.panel {{
    background: {PANEL};
    border: 1px solid {LINE};
    padding: 1rem 1.2rem;
}}
.panel .label {{
    color: {MUTED};
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'Inter', sans-serif;
}}
.panel .value {{
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 0.15rem;
}}

.verdict {{
    border: 1px solid;
    padding: 1.1rem 1.3rem;
    margin-top: 0.8rem;
}}
.verdict-clear {{ border-color: rgba(34,211,238,0.4); background: rgba(34,211,238,0.06); }}
.verdict-alert {{ border-color: rgba(255,59,92,0.45); background: rgba(255,59,92,0.07); }}
.verdict h3 {{ margin: 0 0 0.2rem 0; font-family: 'JetBrains Mono', monospace; }}
.verdict p {{ margin: 0; color: {MUTED}; font-family: 'Inter', sans-serif; font-size: 0.88rem; }}

.log-row {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    padding: 0.5rem 0.7rem;
    border-bottom: 1px solid {LINE};
    display: flex;
    justify-content: space-between;
}}
.log-clear {{ color: {SAFE}; }}
.log-alert {{ color: {ALERT}; }}

hr {{ border-color: {LINE}; }}
footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Load artifacts
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/intrusion_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    le_dict = joblib.load("model/label_encoders.pkl")
    feature_columns = joblib.load("model/feature_columns.pkl")
    importances = pd.read_csv("model/feature_importances.csv")
    return model, scaler, le_dict, feature_columns, importances

@st.cache_data
def load_data():
    train_raw = pd.read_csv("data/nsl_kdd_train_original.csv")
    return train_raw

model, scaler, le_dict, feature_columns, importances_df = load_artifacts()
train_raw = load_data()

if "detection_log" not in st.session_state:
    st.session_state.detection_log = []

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="console-header">
    <h1><span class="blinking-dot"></span>Net<span class="accent">Sentry</span></h1>
    <p>Network intrusion detection console — flags anomalous connections using a Random Forest classifier trained on the NSL-KDD benchmark dataset.</p>
</div>
""", unsafe_allow_html=True)

tab_scan, tab_data, tab_model = st.tabs(["🛰️  Live Scan", "📡  Traffic Data", "🧠  Model Internals"])

# ----------------------------------------------------------------------------
# TAB 1 — Live Scan
# ----------------------------------------------------------------------------
with tab_scan:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.markdown("##### Connection Parameters")
        c1, c2 = st.columns(2)
        with c1:
            protocol_type = st.selectbox("Protocol", list(le_dict['protocol_type'].classes_), index=list(le_dict['protocol_type'].classes_).index('tcp') if 'tcp' in le_dict['protocol_type'].classes_ else 0)
            service = st.selectbox("Service", list(le_dict['service'].classes_), index=list(le_dict['service'].classes_).index('http') if 'http' in le_dict['service'].classes_ else 0)
            flag = st.selectbox("Flag", list(le_dict['flag'].classes_), index=list(le_dict['flag'].classes_).index('SF') if 'SF' in le_dict['flag'].classes_ else 0)
            duration = st.number_input("Duration (s)", 0, 60000, 0)
        with c2:
            src_bytes = st.number_input("Source bytes", 0, 100000, 200)
            dst_bytes = st.number_input("Destination bytes", 0, 100000, 0)
            count = st.number_input("Count (conns to same host)", 0, 500, 5)
            srv_count = st.number_input("Service count", 0, 500, 5)

        c3, c4 = st.columns(2)
        with c3:
            serror_rate = st.slider("Serror rate", 0.0, 1.0, 0.0)
        with c4:
            same_srv_rate = st.slider("Same service rate", 0.0, 1.0, 1.0)

        threshold = st.slider("Detection sensitivity (lower = more sensitive)", 0.1, 0.9, 0.3)
        scan_clicked = st.button("▶ Run Scan", type="primary", use_container_width=True)

    with right:
        st.markdown("##### Threat Assessment")

        if scan_clicked:
            row = {col: 0 for col in feature_columns}
            row['protocol_type'] = le_dict['protocol_type'].transform([protocol_type])[0]
            row['service'] = le_dict['service'].transform([service])[0]
            row['flag'] = le_dict['flag'].transform([flag])[0]
            row['duration'] = duration
            row['src_bytes'] = src_bytes
            row['dst_bytes'] = dst_bytes
            row['count'] = count
            row['srv_count'] = srv_count
            row['serror_rate'] = serror_rate
            row['same_srv_rate'] = same_srv_rate

            input_df = pd.DataFrame([row])[feature_columns]
            input_scaled = scaler.transform(input_df)
            proba = model.predict_proba(input_scaled)[0][1]
            prediction = 1 if proba >= threshold else 0

            gauge_color = ALERT if prediction == 1 else SAFE
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 42, "family": "JetBrains Mono", "color": TEXT}},
                gauge={
                    "shape": "angular",
                    "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"color": MUTED, "family": "JetBrains Mono"}},
                    "bar": {"color": gauge_color, "thickness": 0.25},
                    "bgcolor": PANEL,
                    "borderwidth": 1,
                    "bordercolor": LINE,
                    "threshold": {"line": {"color": AMBER, "width": 3}, "thickness": 0.9, "value": threshold * 100},
                    "steps": [
                        {"range": [0, 30], "color": "rgba(34,211,238,0.08)"},
                        {"range": [30, 70], "color": "rgba(255,176,32,0.08)"},
                        {"range": [70, 100], "color": "rgba(255,59,92,0.10)"},
                    ],
                },
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=260, margin=dict(l=20, r=20, t=20, b=10), font={"color": TEXT},
            )
            st.plotly_chart(fig, use_container_width=True)

            if prediction == 1:
                st.markdown(f"""<div class="verdict verdict-alert"><h3>⚠ INTRUSION FLAGGED</h3>
                <p>Attack confidence {proba:.1%}, above sensitivity threshold {threshold:.0%}.</p></div>""", unsafe_allow_html=True)
                verdict_label = "ALERT"
            else:
                st.markdown(f"""<div class="verdict verdict-clear"><h3>✓ TRAFFIC CLEAR</h3>
                <p>Attack confidence {proba:.1%}, below sensitivity threshold {threshold:.0%}.</p></div>""", unsafe_allow_html=True)
                verdict_label = "CLEAR"

            st.session_state.detection_log.insert(0, {
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "protocol": protocol_type, "service": service, "flag": flag,
                "confidence": f"{proba:.1%}", "verdict": verdict_label,
            })
        else:
            st.info("Set connection parameters on the left and click **Run Scan**.")

        if st.session_state.detection_log:
            st.markdown("##### Detection Log (this session)")
            for entry in st.session_state.detection_log[:8]:
                css_class = "log-alert" if entry["verdict"] == "ALERT" else "log-clear"
                st.markdown(
                    f'<div class="log-row {css_class}">'
                    f'<span>{entry["time"]}</span>'
                    f'<span>{entry["protocol"]}/{entry["service"]}/{entry["flag"]}</span>'
                    f'<span>{entry["confidence"]}</span>'
                    f'<span><b>{entry["verdict"]}</b></span>'
                    f'</div>', unsafe_allow_html=True)
            if st.button("Clear log"):
                st.session_state.detection_log = []
                st.rerun()

# ----------------------------------------------------------------------------
# TAB 2 — Traffic Data
# ----------------------------------------------------------------------------
with tab_data:
    st.markdown("##### NSL-KDD Benchmark Dataset")
    st.markdown(
        "<span style='font-family:Inter,sans-serif;color:#6B7686'>"
        "Standard benchmark for network intrusion detection research — 125,973 labeled "
        "connection records, 41 features per connection, spanning normal traffic and 22 attack types "
        "(DoS, probe, R2L, U2R).</span>", unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)
    for col, label, value in zip(
        [m1, m2, m3, m4],
        ["CONNECTIONS", "FEATURES", "NORMAL", "ATTACKS"],
        [f"{len(train_raw):,}", "41", f"{(train_raw['attack_type']=='normal').sum():,}", f"{(train_raw['attack_type']!='normal').sum():,}"],
    ):
        col.markdown(f"""<div class="panel"><div class="label">{label}</div><div class="value">{value}</div></div>""", unsafe_allow_html=True)

    st.markdown("##### Raw dataset sample (as sourced)")
    st.dataframe(train_raw.head(200), use_container_width=True, height=280)
    st.download_button(
        "⬇ Download full training set (CSV)",
        train_raw.to_csv(index=False).encode("utf-8"),
        file_name="nsl_kdd_train_original.csv",
        mime="text/csv",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Normal vs Attack traffic")
        counts = train_raw['attack_type'].apply(lambda x: 'Normal' if x == 'normal' else 'Attack').value_counts()
        fig3 = px.pie(values=counts.values, names=counts.index, hole=0.55,
                      color=counts.index, color_discrete_map={"Normal": SAFE, "Attack": ALERT})
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": TEXT, "family": "JetBrains Mono"}, height=320,
                            legend={"orientation": "h", "y": -0.1})
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        st.markdown("##### Top attack types")
        top_attacks = train_raw[train_raw['attack_type'] != 'normal']['attack_type'].value_counts().head(8)
        fig4 = px.bar(x=top_attacks.values, y=top_attacks.index, orientation='h', color_discrete_sequence=[ALERT])
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font={"color": TEXT, "family": "JetBrains Mono"}, height=320,
                            yaxis={"categoryorder": "total ascending", "title": ""}, xaxis={"title": "Connections"})
        st.plotly_chart(fig4, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3 — Model Internals
# ----------------------------------------------------------------------------
with tab_model:
    st.markdown("##### Model Performance — the honest version")
    st.markdown(
        "<span style='font-family:Inter,sans-serif;color:#6B7686'>"
        "Same-distribution test splits on NSL-KDD are notoriously easy — attack patterns repeat "
        "between train and test. The numbers below separate that from performance on the official "
        "held-out test set, which contains attack types never seen during training."
        "</span>", unsafe_allow_html=True
    )

    st.markdown("**Same-distribution split (misleadingly optimistic)**")
    m1, m2 = st.columns(2)
    m1.markdown(f"""<div class="panel"><div class="label">ACCURACY</div><div class="value">99.9%</div></div>""", unsafe_allow_html=True)
    m2.markdown(f"""<div class="panel"><div class="label">AUC</div><div class="value">1.000</div></div>""", unsafe_allow_html=True)

    st.markdown("**Official held-out test set — unseen attacks (default threshold 0.5)**")
    m3, m4, m5 = st.columns(3)
    m3.markdown(f"""<div class="panel"><div class="label">ACCURACY</div><div class="value">77.7%</div></div>""", unsafe_allow_html=True)
    m4.markdown(f"""<div class="panel"><div class="label">RECALL</div><div class="value">63.0%</div></div>""", unsafe_allow_html=True)
    m5.markdown(f"""<div class="panel"><div class="label">AUC</div><div class="value">0.962</div></div>""", unsafe_allow_html=True)

    st.markdown("**Threshold-tuned for recall (threshold = 0.3)**")
    m6, m7, m8 = st.columns(3)
    m6.markdown(f"""<div class="panel"><div class="label">ACCURACY</div><div class="value">82.5%</div></div>""", unsafe_allow_html=True)
    m7.markdown(f"""<div class="panel"><div class="label">RECALL</div><div class="value">71.6%</div></div>""", unsafe_allow_html=True)
    m8.markdown(f"""<div class="panel"><div class="label">PRECISION</div><div class="value">96.7%</div></div>""", unsafe_allow_html=True)

    st.markdown("##### Feature importance")
    fig5 = px.bar(importances_df.head(15), x="importance", y="feature", orientation="h", color_discrete_sequence=[SAFE])
    fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": TEXT, "family": "JetBrains Mono"},
                        height=440, yaxis={"categoryorder": "total ascending", "title": ""}, xaxis={"title": "Importance"})
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown(f"""
    <div style="font-family:Inter,sans-serif; color:{MUTED}; font-size:0.9rem; line-height:1.6;">
    <b style="color:{TEXT}">Reading this honestly:</b><br>
    • <code>src_bytes</code>, <code>same_srv_rate</code>, and connection-error rates dominate — high error
    rates and unusual byte counts are classic DoS/probe signatures.<br>
    • Lowering the detection threshold trades some precision for meaningfully better recall — worth it
    here, since a missed attack is costlier than an extra alert to investigate.<br>
    • This is a benchmark-dataset model; a production IDS would need continuous retraining against
    live traffic, since attack patterns evolve.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"<span style='font-family:Inter,sans-serif;color:{MUTED};font-size:0.85rem'>Built with scikit-learn + Streamlit · Dataset: NSL-KDD (a refined benchmark of the classic KDD Cup 1999 dataset)</span>", unsafe_allow_html=True)
