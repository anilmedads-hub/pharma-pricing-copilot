"""
streamlit_app.py
================
Streamlit front-end for the Pharma Pricing Copilot.

Four interactive tabs:

1. Dashboard         — KPIs, WAC trends, GTN heatmap, price benchmarks
2. Anomaly Explorer  — Filterable anomaly table, severity charts, scatter
3. Regulatory Q&A   — RAG-backed plain-English regulatory questions
4. AI Copilot       — Streaming PharmaPricingAgent with tool-use transparency

Run
---
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.agent import PharmaPricingAgent
from src.anomaly_detection import AnomalyDetectionEngine
from src.data_simulator import PricingDataSimulator
from src.rag_pipeline import RAGPipeline
from src.utils import fmt_usd, get_logger

load_dotenv()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Pharma Pricing Copilot",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    .metric-card h2 { font-size: 2rem; margin: 0; color: #fff; }
    .metric-card p  { font-size: 0.85rem; opacity: 0.85; margin: 4px 0 0; }
    .severity-high   { color: #ef553b; font-weight: bold; }
    .severity-medium { color: #ffa15a; font-weight: bold; }
    .severity-low    { color: #19d3f3; font-weight: bold; }
    .stChatMessage { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------


def _init_state() -> None:
    defaults = {
        "data_loaded": False,
        "transactions": None,
        "anomalies": None,
        "agent": None,
        "chat_messages": [],
        "rag_messages": [],
        "rag_history": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 💊 Pharma Pricing Copilot")
        st.markdown("---")

        st.markdown("### 🔑 API Configuration")
        api_key = st.text_input(
            "Anthropic API Key",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            type="password",
            help="Required for AI Copilot and Regulatory Q&A.",
        )
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key

        st.markdown("---")
        st.markdown("### ⚙️ Simulation Parameters")
        seed = st.number_input("Random seed", value=42, step=1)
        n_rows = st.slider("Transactions to generate", 5_000, 20_000, 15_000, 1_000)
        anomaly_rate = st.slider("Anomaly injection rate", 0.01, 0.20, 0.07, 0.01,
                                 format="%.2f")

        if st.button("🚀 Generate & Load Data", type="primary", use_container_width=True):
            with st.spinner("Simulating pharmaceutical pricing data…"):
                sim = PricingDataSimulator(
                    seed=int(seed),
                    n_rows=n_rows,
                    anomaly_rate=anomaly_rate,
                )
                txn_df = sim.generate()
                st.session_state.transactions = txn_df

            with st.spinner("Running ensemble anomaly detection…"):
                engine = AnomalyDetectionEngine()
                anomaly_df = engine.detect_all()
                anomaly_df.to_csv("data/processed/anomaly_results.csv", index=False)
                st.session_state.anomalies = anomaly_df

            # Fresh agent (re-created on each data load so it picks up new files)
            st.session_state.agent = None
            st.session_state.chat_messages = []
            st.session_state.data_loaded = True
            st.success(
                f"✅ Loaded {len(txn_df):,} transactions — "
                f"{len(anomaly_df):,} anomalies detected"
            )

        st.markdown("---")
        if st.session_state.data_loaded:
            txn = st.session_state.transactions
            anom = st.session_state.anomalies
            st.markdown("### 📊 Dataset Summary")
            st.metric("Transactions", f"{len(txn):,}")
            st.metric("Unique Drugs", txn["drug_id"].nunique())
            st.metric("Unique Pharmacies", txn["pharmacy_id"].nunique())
            st.metric("Anomalies", f"{len(anom):,}")
            st.metric(
                "High Severity",
                f"{(anom['severity'] == 'high').sum():,}",
            )

        st.markdown("---")
        st.caption("Powered by Claude claude-sonnet-4-6 · Pharma Pricing Copilot v2.0")


_render_sidebar()

# ---------------------------------------------------------------------------
# Gate: require data before showing tabs
# ---------------------------------------------------------------------------

st.title("💊 Pharma Pricing Copilot")

if not st.session_state.data_loaded:
    st.info("👈 Use the sidebar to generate synthetic pricing data, then explore the tabs.")
    st.markdown("""
    ### What this tool does
    | Tab | Description |
    |---|---|
    | **📊 Dashboard** | WAC trends, GTN analysis, price benchmarks across drug classes |
    | **🔍 Anomaly Explorer** | Drill into detected anomalies with filtering and charts |
    | **📚 Regulatory Q&A** | RAG-powered answers to Medicaid, 340B, and Medicare questions |
    | **🤖 AI Copilot** | Streaming conversational agent with 6 built-in data tools |
    """)
    st.stop()

txn: pd.DataFrame = st.session_state.transactions
anom: pd.DataFrame = st.session_state.anomalies

tab_dash, tab_anom, tab_rag, tab_copilot = st.tabs(
    ["📊 Dashboard", "🔍 Anomaly Explorer", "📚 Regulatory Q&A", "🤖 AI Copilot"]
)

# ===========================================================================
# Tab 1 — Dashboard
# ===========================================================================

with tab_dash:
    st.header("Pricing Dashboard")

    # ---- KPI row -----------------------------------------------------------
    total_txn    = len(txn)
    n_drugs      = txn["drug_id"].nunique()
    avg_wac      = txn["wac_price"].mean()
    anomaly_rate_actual = len(anom) / total_txn * 100
    avg_margin   = txn["margin_percent"].mean()
    high_sev     = int((anom["severity"] == "high").sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Transactions", f"{total_txn:,}")
    c2.metric("Unique Drugs", n_drugs)
    c3.metric("Avg WAC", fmt_usd(avg_wac))
    c4.metric("Avg Margin %", f"{avg_margin:.1f}%")
    c5.metric("Anomaly Rate", f"{anomaly_rate_actual:.1f}%")
    c6.metric("High-Severity", high_sev, delta=None)

    st.markdown("---")

    # ---- WAC Trends (top 8 drugs by volume) --------------------------------
    st.subheader("WAC Price Trends — Top 8 Drugs by Transaction Volume")
    top8_drugs = txn["drug_name"].value_counts().head(8).index.tolist()
    trend_df = txn[txn["drug_name"].isin(top8_drugs)].copy()
    trend_df["month"] = pd.to_datetime(trend_df["transaction_date"]).dt.to_period("M").astype(str)
    trend_monthly = (
        trend_df.groupby(["month", "drug_name"])["wac_price"]
        .mean()
        .reset_index()
    )
    fig_trend = px.line(
        trend_monthly,
        x="month",
        y="wac_price",
        color="drug_name",
        title="Monthly Average WAC (USD)",
        labels={"wac_price": "WAC (USD)", "month": "Month", "drug_name": "Drug"},
        template="plotly_white",
    )
    fig_trend.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_trend, use_container_width=True)

    col_left, col_right = st.columns(2)

    # ---- GTN Discount Heatmap by Drug Class --------------------------------
    with col_left:
        st.subheader("Gross-to-Net Discount % by Drug Class")
        txn["gtn_pct"] = (txn["wac_price"] - txn["gtn_price"]) / txn["wac_price"] * 100
        txn["quarter"] = pd.to_datetime(txn["transaction_date"]).dt.to_period("Q").astype(str)
        heat_data = (
            txn.groupby(["drug_class", "quarter"])["gtn_pct"]
            .mean()
            .reset_index()
        )
        heat_pivot = heat_data.pivot(index="drug_class", columns="quarter", values="gtn_pct")
        fig_heat = px.imshow(
            heat_pivot,
            color_continuous_scale="RdYlGn_r",
            title="Avg GTN Discount % (red = deeper discount)",
            labels={"color": "GTN %"},
            template="plotly_white",
            aspect="auto",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ---- Price Benchmarks by Drug Class ------------------------------------
    with col_right:
        st.subheader("Price Benchmarks by Drug Class")
        bench = (
            txn.groupby("drug_class")[["wac_price", "asp_price", "amp_price", "price_340b"]]
            .mean()
            .reset_index()
            .melt(id_vars="drug_class", var_name="Benchmark", value_name="Avg Price (USD)")
        )
        fig_bench = px.bar(
            bench,
            x="drug_class",
            y="Avg Price (USD)",
            color="Benchmark",
            barmode="group",
            title="Average WAC / ASP / AMP / 340B by Drug Class",
            template="plotly_white",
            labels={"drug_class": "Drug Class"},
        )
        st.plotly_chart(fig_bench, use_container_width=True)

    # ---- Anomaly Severity Distribution + Top Anomalous Drugs ---------------
    st.markdown("---")
    c_sev, c_top = st.columns(2)

    with c_sev:
        st.subheader("Anomaly Severity Distribution")
        sev_counts = anom["severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        color_map = {"high": "#EF553B", "medium": "#FFA15A", "low": "#19D3F3"}
        fig_sev = px.pie(
            sev_counts,
            names="Severity",
            values="Count",
            color="Severity",
            color_discrete_map=color_map,
            title="Anomalies by Severity",
            hole=0.4,
            template="plotly_white",
        )
        st.plotly_chart(fig_sev, use_container_width=True)

    with c_top:
        st.subheader("Top 10 Drugs by Anomaly Count")
        top_drugs = (
            anom.groupby("drug_name")
            .size()
            .reset_index(name="Anomaly Count")
            .nlargest(10, "Anomaly Count")
        )
        fig_top = px.bar(
            top_drugs,
            x="Anomaly Count",
            y="drug_name",
            orientation="h",
            title="Most Flagged Drugs",
            template="plotly_white",
            color="Anomaly Count",
            color_continuous_scale="Reds",
        )
        fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_top, use_container_width=True)

    # ---- Margin Distribution -----------------------------------------------
    st.subheader("Margin % Distribution by Drug Class")
    fig_box = px.box(
        txn,
        x="drug_class",
        y="margin_percent",
        color="drug_class",
        title="Margin % Distribution (WAC → Actual Price)",
        template="plotly_white",
        labels={"margin_percent": "Margin %", "drug_class": "Drug Class"},
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ===========================================================================
# Tab 2 — Anomaly Explorer
# ===========================================================================

with tab_anom:
    st.header("Anomaly Explorer")

    if anom.empty:
        st.success("No anomalies detected.")
    else:
        # ---- Filters -------------------------------------------------------
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sev_filter = st.multiselect(
                "Severity",
                options=["high", "medium", "low"],
                default=["high", "medium"],
            )
        with fc2:
            type_filter = st.multiselect(
                "Anomaly Type",
                options=sorted(anom["anomaly_type"].dropna().unique()),
                default=[],
            )
        with fc3:
            drug_filter = st.selectbox(
                "Drug",
                options=["All"] + sorted(anom["drug_name"].unique().tolist()),
            )

        display = anom.copy()
        if sev_filter:
            display = display[display["severity"].isin(sev_filter)]
        if type_filter:
            display = display[display["anomaly_type"].isin(type_filter)]
        if drug_filter != "All":
            display = display[display["drug_name"] == drug_filter]

        st.markdown(f"**{len(display):,} anomalies** matching filters")

        # ---- Table ---------------------------------------------------------
        st.dataframe(
            display[[
                "drug_name", "drug_class", "pharmacy_name", "state",
                "anomaly_type", "severity", "detection_method",
                "anomaly_score", "description", "recommended_action",
                "transaction_date",
            ]]
            .sort_values(["severity", "anomaly_score"], ascending=[True, False])
            .reset_index(drop=True),
            use_container_width=True,
            height=350,
        )

        st.markdown("---")
        ca, cb = st.columns(2)

        # ---- Anomaly Type Breakdown ----------------------------------------
        with ca:
            st.subheader("Anomaly Type Breakdown")
            type_counts = (
                display["anomaly_type"].value_counts().reset_index()
            )
            type_counts.columns = ["Type", "Count"]
            fig_type = px.bar(
                type_counts,
                x="Type",
                y="Count",
                color="Count",
                color_continuous_scale="Oranges",
                title="Flagged Transactions by Anomaly Type",
                template="plotly_white",
            )
            fig_type.update_xaxes(tickangle=30)
            st.plotly_chart(fig_type, use_container_width=True)

        # ---- Detection Method Coverage -------------------------------------
        with cb:
            st.subheader("Detection Method Coverage")
            # Each row may list multiple methods separated by " | "
            methods = (
                display["detection_method"]
                .str.split(" | ")
                .explode()
                .str.strip()
                .value_counts()
                .reset_index()
            )
            methods.columns = ["Method", "Count"]
            fig_method = px.bar(
                methods,
                x="Method",
                y="Count",
                color="Count",
                color_continuous_scale="Blues",
                title="How Each Anomaly Was Detected",
                template="plotly_white",
            )
            fig_method.update_xaxes(tickangle=30)
            st.plotly_chart(fig_method, use_container_width=True)

        # ---- WAC vs Actual Price scatter -----------------------------------
        st.subheader("WAC vs Actual Price — Anomalies Highlighted")
        scatter_df = display.copy()
        scatter_df["Severity"] = scatter_df["severity"].str.capitalize()
        fig_scatter = px.scatter(
            scatter_df,
            x="wac_price",
            y="actual_price",
            color="Severity",
            color_discrete_map={"High": "#EF553B", "Medium": "#FFA15A", "Low": "#19D3F3"},
            hover_data=["drug_name", "anomaly_type", "description"],
            title="WAC Price vs Actual Transaction Price",
            labels={"wac_price": "WAC (USD)", "actual_price": "Actual Price (USD)"},
            template="plotly_white",
            opacity=0.7,
        )
        # diagonal reference line
        max_val = max(scatter_df["wac_price"].max(), scatter_df["actual_price"].max())
        fig_scatter.add_shape(
            type="line", x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(color="grey", dash="dash", width=1),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # ---- Anomalies over time -------------------------------------------
        st.subheader("Anomaly Volume Over Time")
        time_df = display.copy()
        time_df["month"] = pd.to_datetime(time_df["transaction_date"]).dt.to_period("M").astype(str)
        time_counts = (
            time_df.groupby(["month", "severity"])
            .size()
            .reset_index(name="Count")
        )
        fig_time = px.area(
            time_counts,
            x="month",
            y="Count",
            color="severity",
            color_discrete_map={"high": "#EF553B", "medium": "#FFA15A", "low": "#19D3F3"},
            title="Anomaly Volume by Month and Severity",
            template="plotly_white",
            labels={"month": "Month", "severity": "Severity"},
        )
        st.plotly_chart(fig_time, use_container_width=True)

# ===========================================================================
# Tab 3 — Regulatory Q&A (RAG)
# ===========================================================================

with tab_rag:
    st.header("Regulatory Q&A")
    st.caption(
        "Ask plain-English questions about Medicaid Best Price, 340B ceiling prices, "
        "ASP/AMP compliance, GTN rebate structures, and more. "
        "Answers are grounded in the built-in regulatory knowledge base."
    )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("Enter your Anthropic API Key in the sidebar to use this feature.")
    else:
        # Example questions
        if not st.session_state.rag_messages:
            st.markdown("**Try asking:**")
            examples = [
                "What is the Medicaid Best Price rule?",
                "How is the 340B ceiling price calculated?",
                "What triggers an ASP/AMP violation?",
                "Explain the gross-to-net bubble problem.",
            ]
            ex_cols = st.columns(len(examples))
            for i, ex in enumerate(examples):
                if ex_cols[i].button(ex, key=f"rag_ex_{i}"):
                    st.session_state._rag_pending = ex
                    st.rerun()

        for msg in st.session_state.rag_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        rag_pending = st.session_state.pop("_rag_pending", None)
        rag_input = st.chat_input("Ask a regulatory question…", key="rag_input") or rag_pending

        if rag_input:
            st.session_state.rag_messages.append({"role": "user", "content": rag_input})
            with st.chat_message("user"):
                st.markdown(rag_input)

            with st.chat_message("assistant"):
                with st.spinner("Searching regulatory knowledge base…"):
                    try:
                        rag = RAGPipeline()
                        answer = rag.answer(
                            rag_input,
                            conversation_history=st.session_state.rag_history,
                        )
                    except Exception as exc:
                        answer = (
                            f"Error retrieving answer: {exc}. "
                            "Ensure your API key is valid."
                        )
                st.markdown(answer)

            st.session_state.rag_messages.append({"role": "assistant", "content": answer})
            st.session_state.rag_history.append({"role": "user", "content": rag_input})
            st.session_state.rag_history.append({"role": "assistant", "content": answer})
            # Keep last 10 exchanges
            st.session_state.rag_history = st.session_state.rag_history[-20:]

# ===========================================================================
# Tab 4 — AI Copilot (PharmaPricingAgent)
# ===========================================================================

with tab_copilot:
    st.header("AI Pricing Copilot")
    st.caption(
        "Chat with a Claude-powered agent that retrieves anomaly data, queries the "
        "pricing database, profiles pharmacy risk, analyses causal trends, and "
        "searches regulatory documents — all in one conversation."
    )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("Enter your Anthropic API Key in the sidebar to use this feature.")
    else:
        col_reset, col_summary, _ = st.columns([1, 1, 5])
        with col_reset:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_messages = []
                if st.session_state.agent:
                    st.session_state.agent.reset_conversation()
                st.rerun()
        with col_summary:
            if st.button("📋 Summary") and st.session_state.agent:
                s = st.session_state.agent.get_conversation_summary()
                st.toast(
                    f"Turns: {s['total_turns']} | Tools: {', '.join(s['tools_called']) or 'none'}"
                )

        # Display history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Suggested starters
        if not st.session_state.chat_messages:
            st.markdown("**Suggested questions:**")
            suggestions = [
                "What are the top 5 high-severity anomalies?",
                "Profile pharmacy risk for PH0042",
                "What is the 340B breach pattern for Revlimid?",
                "Show me a causal summary for DRG025",
                "Explain the regulatory rules around Best Price",
            ]
            s_cols = st.columns(len(suggestions))
            for i, sug in enumerate(suggestions):
                if s_cols[i].button(sug, key=f"sug_{i}"):
                    st.session_state._pending_msg = sug
                    st.rerun()

        pending = st.session_state.pop("_pending_msg", None)
        user_input = st.chat_input("Ask about drug pricing…", key="agent_input") or pending

        if user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""

                try:
                    if st.session_state.agent is None:
                        st.session_state.agent = PharmaPricingAgent()

                    agent: PharmaPricingAgent = st.session_state.agent

                    for chunk in agent.chat_streaming(user_input):
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")

                    placeholder.markdown(full_response)

                except Exception as exc:
                    full_response = f"⚠️ Agent error: {exc}"
                    placeholder.error(full_response)
                    logger.exception("Agent error in Streamlit chat")

            st.session_state.chat_messages.append(
                {"role": "assistant", "content": full_response}
            )
