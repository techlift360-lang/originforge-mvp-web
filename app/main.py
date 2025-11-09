"""
OriginForge — Policy Sandbox (Streamlit Front-End)

This module defines the full Streamlit app for:
- Single-scenario simulations (with presets + custom sliders)
- Scenario comparison (A vs B)
- Run history (session)
- Developer view (JSON payload for the last single run)

Backed by:
- world.World            -> simulation engine
- utils.export_pdf_bytes -> PDF policy brief generation
"""

from __future__ import annotations

import json
from typing import Dict, Any, List, Optional

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from world import World
from utils import export_pdf_bytes

# ---------------------------
# Page Config & Session Setup
# ---------------------------

st.set_page_config(
    page_title="OriginForge — Policy Sandbox",
    page_icon="🌍",
    layout="wide",
)

# Initialize session state keys
if "run_history" not in st.session_state:
    st.session_state["run_history"]: List[Dict[str, Any]] = []

if "last_single_run" not in st.session_state:
    st.session_state["last_single_run"]: Optional[Dict[str, Any]] = None

# ---------------------------
# Preset Scenario Definitions
# ---------------------------

SCENARIOS: Dict[str, Optional[Dict[str, Any]]] = {
    "Custom (use sliders)": None,
    "High UBI Safety Net": {
        "tax": 0.28,
        "ubi": 0.25,
        "edu": 0.07,
        "cap": 0.15,
        "regime": "democracy",
    },
    "High Tax & Education": {
        "tax": 0.35,
        "ubi": 0.12,
        "edu": 0.10,
        "cap": 0.20,
        "regime": "democracy",
    },
    "Low Tax / High Growth": {
        "tax": 0.15,
        "ubi": 0.05,
        "edu": 0.04,
        "cap": 0.05,
        "regime": "autocracy",
    },
}

# ---------------------------
# Helper Functions
# ---------------------------

def clamp_float(value: Any, lo: float, hi: float, fallback: float) -> float:
    """Convert to float and clamp to [lo, hi]; fallback on error."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(lo, min(hi, x))


def clamp_int(value: Any, lo: int, hi: int, fallback: int) -> int:
    """Convert to int and clamp to [lo, hi]; fallback on error."""
    try:
        x = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(lo, min(hi, x))


@st.cache_data(show_spinner=False)
def run_world_with_params_cached(
    tax: float,
    ubi: float,
    edu: float,
    cap: float,
    regime: str,
    ticks: int,
) -> pd.DataFrame:
    """
    Cached simulation runner.

    Same parameters + ticks -> instantly returns cached results.
    """
    # Basic safety guard
    ticks = max(1, min(int(ticks), 5000))

    world = World(
        tax_rate=tax,
        ubi_rate=ubi,
        education_spend=edu,
        resource_cap=cap,
        regime=regime,
    )
    df = world.run(ticks=ticks)
    return df


def get_params_for_scenario(name: str, sliders: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve the effective parameters, either from a preset scenario
    or from the current slider values (Custom mode).
    """
    if SCENARIOS.get(name) is None:
        # Custom mode: use sliders
        return {
            "tax": float(sliders["tax"]),
            "ubi": float(sliders["ubi"]),
            "edu": float(sliders["edu"]),
            "cap": float(sliders["cap"]),
            "regime": str(sliders["regime"]),
        }
    return SCENARIOS[name].copy()  # type: ignore[return-value]


def pct_change(new: float, old: float) -> float:
    """Safe percent change helper."""
    if old == 0:
        return 0.0
    return 100.0 * (new - old) / abs(old)


def describe_single_run(df: pd.DataFrame) -> str:
    """
    Create a human-readable summary for a single scenario run.
    Uses first vs last row to describe trends.
    """
    if df.empty or len(df) < 2:
        return "Not enough data to summarize."

    first = df.iloc[0]
    last = df.iloc[-1]

    gdp_change = pct_change(float(last["gdp"]), float(first["gdp"]))
    gini_change = pct_change(float(last["gini_proxy"]), float(first["gini_proxy"]))
    stab_change = pct_change(float(last["stability"]), float(first["stability"]))

    lines: List[str] = []

    # GDP
    if gdp_change > 2:
        lines.append(f"• GDP grew by about **{gdp_change:.1f}%** over the simulation.")
    elif gdp_change < -2:
        lines.append(f"• GDP decreased by about **{abs(gdp_change):.1f}%** over the simulation.")
    else:
        lines.append("• GDP stayed relatively **stable** over the simulation.")

    # Inequality (Gini)
    if gini_change < -1:
        lines.append(f"• Inequality (Gini) **decreased** by roughly {abs(gini_change):.1f}%.")
    elif gini_change > 1:
        lines.append(f"• Inequality (Gini) **increased** by roughly {gini_change:.1f}%.")
    else:
        lines.append("• Inequality (Gini) remained **roughly constant**.")

    # Stability
    if stab_change > 2:
        lines.append(f"• Social stability **improved**, rising by about {stab_change:.1f}%.")
    elif stab_change < -2:
        lines.append(f"• Social stability **declined**, falling by about {abs(stab_change):.1f}%.")
    else:
        lines.append("• Social stability stayed **fairly stable**.")

    return "\n".join(lines)


def build_policy_brief_lines(
    scenario_name: str,
    params: Dict[str, Any],
    ticks: int,
    df: pd.DataFrame,
    insight_md: str,
) -> List[str]:
    """
    Convert run info + insight into a list of plain-text lines for PDF export.
    """
    lines: List[str] = []
    lines.append(f"Scenario: {scenario_name}")
    lines.append(f"Ticks simulated: {int(ticks)}")
    lines.append("")
    lines.append("Policy Parameters:")
    lines.append(f"- Tax rate: {params['tax']:.3f}")
    lines.append(f"- UBI (fraction of median income): {params['ubi']:.3f}")
    lines.append(f"- Education (GDP share): {params['edu']:.3f}")
    lines.append(f"- Resource cap: {params['cap']:.3f}")
    lines.append(f"- Regime: {params['regime']}")
    lines.append("")

    if not df.empty:
        last = df.iloc[-1]
        lines.append("Final Metrics:")
        lines.append(f"- GDP (proxy): {float(last['gdp']):.2f}")
        lines.append(f"- Inequality (Gini proxy): {float(last['gini_proxy']):.3f}")
        lines.append(f"- Stability: {float(last['stability']):.3f}")
        lines.append(f"- Innovation: {float(last['innovation']):.3f}")
        lines.append(f"- Emissions: {float(last['emissions']):.3f}")
        lines.append("")
    else:
        lines.append("No metrics available (empty run).")
        lines.append("")

    lines.append("Quick Insight:")
    # Convert markdown bullets to plain text for PDF
    for line in insight_md.splitlines():
        plain = line.replace("• ", "- ").replace("**", "")
        lines.append(plain)

    return lines


def add_run_to_history(name: str, params: Dict[str, Any], df: pd.DataFrame) -> None:
    """
    Store a compact summary of the run in session history.
    """
    if df.empty:
        return

    last = df.iloc[-1]
    st.session_state["run_history"].insert(
        0,
        {
            "Scenario name": name,
            "Tax": round(params["tax"], 3),
            "UBI": round(params["ubi"], 3),
            "Edu": round(params["edu"], 3),
            "Cap": round(params["cap"], 3),
            "Regime": params["regime"],
            "Final GDP": round(float(last["gdp"]), 1),
            "Final Gini": round(float(last["gini_proxy"]), 3),
            "Final Stability": round(float(last["stability"]), 3),
        },
    )
    # keep only last 10
    st.session_state["run_history"] = st.session_state["run_history"][:10]


# ---------------------------
# Header & Intro Section
# ---------------------------

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        """
        ## 🌍 OriginForge — Policy Sandbox  
        <span style="font-size: 14px; color: #9CA3AF;">
        Interactively explore how tax, UBI, education, and resource policies shape a virtual society over time.
        </span>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        """
        <div style="text-align:right; font-size: 12px; color:#6B7280;">
        v1.0 • Simulation sandbox (educational & exploratory)
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

# Quick 3-step cards
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div style="border-radius: 14px; padding: 12px 14px; background-color: #020617; border: 1px solid #1E293B;">
          <div style="font-size: 12px; color:#64748B; text-transform: uppercase; letter-spacing: .08em;">Step 1</div>
          <div style="font-size: 14px; font-weight: 600; margin-top: 4px;">Choose policies</div>
          <div style="font-size: 12px; color:#9CA3AF; margin-top: 4px;">Pick a preset or tune tax, UBI, education, and resource caps.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div style="border-radius: 14px; padding: 12px 14px; background-color: #020617; border: 1px solid #1E293B;">
          <div style="font-size: 12px; color:#64748B; text-transform: uppercase; letter-spacing: .08em;">Step 2</div>
          <div style="font-size: 14px; font-weight: 600; margin-top: 4px;">Run the simulation</div>
          <div style="font-size: 12px; color:#9CA3AF; margin-top: 4px;">See how GDP, inequality, stability, innovation, and emissions evolve.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div style="border-radius: 14px; padding: 12px 14px; background-color: #020617; border: 1px solid #1E293B;">
          <div style="font-size: 12px; color:#64748B; text-transform: uppercase; letter-spacing: .08em;">Step 3</div>
          <div style="font-size: 14px; font-weight: 600; margin-top: 4px;">Compare & export</div>
          <div style="font-size: 12px; color:#9CA3AF; margin-top: 4px;">Compare scenarios side-by-side and export CSV, PDF, or JSON.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

with st.expander("How the model works", expanded=False):
    st.markdown(
        """
        - The world contains households, firms, and a government.  
        - Policies influence income, redistribution, innovation, emissions, and stability.  
        - Each **tick** represents a time step (e.g., month or year) where agents interact.  
        - Metrics are proxies, calibrated for exploration and teaching rather than precise forecasts.
        """.strip()
    )

with st.expander("About & disclaimer", expanded=False):
    st.markdown(
        """
        **OriginForge** is a policy *sandbox* that models a simplified virtual economy.  
        It is designed for **education, experimentation, and scenario exploration** —  
        not as a precise forecast of any specific real-world country.
        """.strip()
    )

st.markdown("---")

# ---------------------------
# Sidebar: Global Controls
# ---------------------------

with st.sidebar:
    st.markdown("### 🎛 Scenario Controls")

    scenario_name_input = st.text_input(
        "Scenario name",
        value="My Scenario",
        help="This name appears in the policy brief, run history, and JSON payload.",
    )

    scenario_preset = st.selectbox(
        "Preset",
        list(SCENARIOS.keys()),
        help="Choose a preset, or 'Custom (use sliders)' to control everything manually.",
        key="single_scenario",
    )

    st.markdown("#### Policy sliders")
    st.caption("Used when preset is **Custom (use sliders)**.")

    tax = st.slider(
        "Tax rate",
        0.0,
        0.50,
        0.20,
        0.01,
        help="Higher tax slightly dampens output but can support redistribution.",
        key="single_tax",
    )
    ubi = st.slider(
        "UBI (fraction of median income)",
        0.0,
        0.30,
        0.10,
        0.01,
        help="Universal Basic Income as a fraction of median wage.",
        key="single_ubi",
    )
    edu = st.slider(
        "Education spend (GDP share)",
        0.0,
        0.10,
        0.05,
        0.01,
        help="Public education investment as a share of GDP.",
        key="single_edu",
    )
    cap = st.slider(
        "Resource cap (reduction)",
        0.0,
        0.40,
        0.20,
        0.01,
        help="Stronger caps reduce emissions but can also reduce output.",
        key="single_cap",
    )
    regime = st.selectbox(
        "Regime",
        ["democracy", "autocracy"],
        help="Regime affects baseline stability in this model.",
        key="single_regime",
    )

    ticks = st.number_input(
        "Simulation ticks",
        min_value=50,
        max_value=5000,
        value=200,
        step=50,
        help="Higher values simulate longer time horizons.",
        key="single_ticks",
    )

    st.markdown("---")
    st.markdown("#### 💾 Save / Load config")

    current_config = {
        "name": scenario_name_input,
        "scenario": scenario_preset,
        "tax": tax,
        "ubi": ubi,
        "edu": edu,
        "cap": cap,
        "regime": regime,
        "ticks": int(ticks),
    }
    config_bytes = json.dumps(current_config, indent=2).encode("utf-8")
    st.download_button(
        label="💾 Download config (JSON)",
        data=config_bytes,
        file_name="originforge_scenario.json",
        mime="application/json",
    )

    uploaded = st.file_uploader(
        "Load config (JSON)",
        type=["json"],
        key="config_uploader",
    )
    if uploaded is not None:
        try:
            loaded_cfg = json.loads(uploaded.read().decode("utf-8"))
            # minimal validation
            if not isinstance(loaded_cfg, dict):
                raise ValueError("Config must be a JSON object.")

            loaded_scenario = loaded_cfg.get("scenario", scenario_preset)
            if loaded_scenario not in SCENARIOS:
                st.warning(
                    "Scenario name in config is not recognized. "
                    "Falling back to 'Custom (use sliders)'."
                )
                loaded_scenario = "Custom (use sliders)"

            st.session_state["single_scenario"] = loaded_scenario

            st.session_state["single_tax"] = clamp_float(
                loaded_cfg.get("tax", tax), 0.0, 0.5, float(tax)
            )
            st.session_state["single_ubi"] = clamp_float(
                loaded_cfg.get("ubi", ubi), 0.0, 0.3, float(ubi)
            )
            st.session_state["single_edu"] = clamp_float(
                loaded_cfg.get("edu", edu), 0.0, 0.1, float(edu)
            )
            st.session_state["single_cap"] = clamp_float(
                loaded_cfg.get("cap", cap), 0.0, 0.4, float(cap)
            )
            st.session_state["single_regime"] = str(
                loaded_cfg.get("regime", regime)
            )
            st.session_state["single_ticks"] = clamp_int(
                loaded_cfg.get("ticks", ticks), 50, 5000, int(ticks)
            )
            st.success("Config loaded. Sliders and preset will update on next rerun.")
        except Exception as e:
            st.error(f"Could not load config: {e}")

# ---------------------------
# Tabs: Main Sections
# ---------------------------

tab_single, tab_compare, tab_history, tab_dev = st.tabs(
    ["Single Scenario", "Compare Scenarios", "Run History", "Developer View"]
)

# ==============
# TAB 1: SINGLE
# ==============
with tab_single:
    st.markdown(
        """
        #### 🎯 Single Scenario  
        Configure a single policy setup and see how the virtual society evolves over time.
        """.strip()
    )

    sliders_single = {
        "tax": tax,
        "ubi": ubi,
        "edu": edu,
        "cap": cap,
        "regime": regime,
    }
    params_single = get_params_for_scenario(scenario_preset, sliders_single)

    # Clarify whether sliders are in effect
    if scenario_preset == "Custom (use sliders)":
        st.info("Custom mode: the policy sliders in the left sidebar define this scenario.")
    else:
        st.info(
            f"Preset mode: using **{scenario_preset}**. "
            "Policy sliders are ignored for this run."
        )

    col1, col2 = st.columns([2.2, 1.0])

    run_button = st.button("▶️ Run single scenario", type="primary")
    st.markdown("")

    if run_button:
        with st.spinner("Running simulation..."):
            try:
                df = run_world_with_params_cached(
                    params_single["tax"],
                    params_single["ubi"],
                    params_single["edu"],
                    params_single["cap"],
                    params_single["regime"],
                    int(ticks),
                )
            except Exception as e:
                st.error(f"An error occurred while running the simulation: {e}")
                df = pd.DataFrame()

        if not df.empty:
            add_run_to_history(scenario_name_input, params_single, df)
            st.session_state["last_single_run"] = {
                "name": scenario_name_input,
                "params": params_single,
                "ticks": int(ticks),
                "df": df,
            }

            # --- Left: charts & details ---
            with col1:
                st.subheader("Metrics over time")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["tick"], y=df["gdp"], name="GDP"))
                fig.add_trace(
                    go.Scatter(
                        x=df["tick"],
                        y=df["gini_proxy"],
                        name="Inequality (Gini proxy)",
                    )
                )
                fig.add_trace(
                    go.Scatter(x=df["tick"], y=df["stability"], name="Stability")
                )
                fig.add_trace(
                    go.Scatter(x=df["tick"], y=df["innovation"], name="Innovation")
                )
                fig.add_trace(
                    go.Scatter(x=df["tick"], y=df["emissions"], name="Emissions")
                )
                fig.update_layout(
                    height=450,
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Advanced view: final snapshot", expanded=False):
                    c1, c2 = st.columns(2)
                    last_row = df.iloc[-1]
                    c1.metric("Final GDP", f"{float(last_row['gdp']):.1f}")
                    c1.metric("Final Emissions", f"{float(last_row['emissions']):.1f}")
                    c2.metric("Final Gini", f"{float(last_row['gini_proxy']):.3f}")
                    c2.metric("Final Stability", f"{float(last_row['stability']):.3f}")

                st.markdown("### Parameters used")
                st.write(
                    {
                        "Scenario preset": scenario_preset,
                        "Scenario name": scenario_name_input,
                        "Tax rate": round(params_single["tax"], 3),
                        "UBI (fraction)": round(params_single["ubi"], 3),
                        "Education (GDP share)": round(params_single["edu"], 3),
                        "Resource cap": round(params_single["cap"], 3),
                        "Regime": params_single["regime"],
                        "Ticks": int(ticks),
                    }
                )

            # --- Right: summary, insight, exports ---
            with col2:
                st.subheader("Summary snapshot")
                last = df.iloc[-1].to_dict()
                st.metric("Population", f'{int(last["population"])}')
                st.metric("GDP (proxy)", f'{float(last["gdp"]):.1f}')
                st.metric("Inequality (Gini)", f'{float(last["gini_proxy"]):.3f}')
                st.metric("Stability", f'{float(last["stability"]):.3f}')
                st.metric("Innovation", f'{float(last["innovation"]):.3f}')
                st.metric("Emissions", f'{float(last["emissions"]):.3f}')

                st.markdown("### Quick insight")
                insight_md = describe_single_run(df)
                st.markdown(insight_md)

                st.markdown("### Export results")
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download data (CSV)",
                    data=csv_data,
                    file_name="originforge_single_run.csv",
                    mime="text/csv",
                )

                brief_lines = build_policy_brief_lines(
                    scenario_name=scenario_name_input,
                    params=params_single,
                    ticks=int(ticks),
                    df=df,
                    insight_md=insight_md,
                )
                pdf_bytes = export_pdf_bytes(
                    brief_lines, title="OriginForge Policy Brief"
                )
                st.download_button(
                    label="📄 Download policy brief (PDF)",
                    data=pdf_bytes,
                    file_name="originforge_policy_brief.pdf",
                    mime="application/pdf",
                )
        else:
            st.warning("No data produced; please adjust settings and try again.")

# ===================
# TAB 2: COMPARISON
# ===================
with tab_compare:
    st.markdown(
        """
        #### ⚖️ Compare Scenarios  
        Pick two presets and compare how GDP and inequality evolve under each policy regime.
        """.strip()
    )

    colA, colB = st.columns(2)

    with colA:
        scenario_A = st.selectbox(
            "Scenario A",
            [k for k in SCENARIOS.keys() if k != "Custom (use sliders)"],
            index=0,
            key="scenario_A",
        )

    with colB:
        scenario_B = st.selectbox(
            "Scenario B",
            [k for k in SCENARIOS.keys() if k != "Custom (use sliders)"],
            index=1,
            key="scenario_B",
        )

    ticks_cmp = st.number_input(
        "Simulation ticks (both scenarios)",
        min_value=50,
        max_value=5000,
        value=300,
        step=50,
        help="Both scenarios will run for the same number of ticks.",
        key="compare_ticks",
    )

    run_compare = st.button("▶️ Run comparison")
    st.markdown("")

    if run_compare:
        with st.spinner("Running comparison…"):
            try:
                params_A = SCENARIOS[scenario_A]  # type: ignore[assignment]
                params_B = SCENARIOS[scenario_B]  # type: ignore[assignment]

                df_A = run_world_with_params_cached(
                    params_A["tax"],
                    params_A["ubi"],
                    params_A["edu"],
                    params_A["cap"],
                    params_A["regime"],
                    int(ticks_cmp),
                )
                df_B = run_world_with_params_cached(
                    params_B["tax"],
                    params_B["ubi"],
                    params_B["edu"],
                    params_B["cap"],
                    params_B["regime"],
                    int(ticks_cmp),
                )
            except Exception as e:
                st.error(f"Error running comparison: {e}")
                df_A, df_B = pd.DataFrame(), pd.DataFrame()

        if not df_A.empty and not df_B.empty:
            # GDP comparison
            st.subheader("GDP over time")
            fig_gdp = go.Figure()
            fig_gdp.add_trace(
                go.Scatter(
                    x=df_A["tick"], y=df_A["gdp"], name=f"GDP – {scenario_A}"
                )
            )
            fig_gdp.add_trace(
                go.Scatter(
                    x=df_B["tick"], y=df_B["gdp"], name=f"GDP – {scenario_B}"
                )
            )
            fig_gdp.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_gdp, use_container_width=True)

            # Gini comparison
            st.subheader("Inequality (Gini) over time")
            fig_gini = go.Figure()
            fig_gini.add_trace(
                go.Scatter(
                    x=df_A["tick"],
                    y=df_A["gini_proxy"],
                    name=f"Gini – {scenario_A}",
                )
            )
            fig_gini.add_trace(
                go.Scatter(
                    x=df_B["tick"],
                    y=df_B["gini_proxy"],
                    name=f"Gini – {scenario_B}",
                )
            )
            fig_gini.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_gini, use_container_width=True)

            # Simple text insight
            last_A = df_A.iloc[-1]
            last_B = df_B.iloc[-1]

            gdp_diff = float(last_B["gdp"]) - float(last_A["gdp"])
            gini_diff = float(last_B["gini_proxy"]) - float(last_A["gini_proxy"])

            st.markdown("### Quick comparison insight")
            insight_lines: List[str] = []

            if abs(gdp_diff) < 1e-6:
                insight_lines.append(
                    f"**{scenario_A}** and **{scenario_B}** end with very similar GDP levels."
                )
            elif gdp_diff > 0:
                insight_lines.append(
                    f"**{scenario_B}** ends with **higher GDP** than **{scenario_A}** "
                    f"by {gdp_diff:,.1f} units."
                )
            else:
                insight_lines.append(
                    f"**{scenario_A}** ends with **higher GDP** than **{scenario_B}** "
                    f"by {abs(gdp_diff):,.1f} units."
                )

            if abs(gini_diff) < 1e-6:
                insight_lines.append(
                    f"Inequality (Gini) is also very similar between **{scenario_A}** and **{scenario_B}**."
                )
            elif gini_diff > 0:
                insight_lines.append(
                    f"**{scenario_B}** ends with **higher inequality (Gini)** than "
                    f"**{scenario_A}** by {gini_diff:.3f}."
                )
            else:
                insight_lines.append(
                    f"**{scenario_A}** ends with **higher inequality (Gini)** than "
                    f"**{scenario_B}** by {abs(gini_diff):.3f}."
                )

            st.write("\n\n".join(insight_lines))

            # Combined CSV export
            df_A_copy = df_A.copy()
            df_B_copy = df_B.copy()
            df_A_copy["scenario"] = scenario_A
            df_B_copy["scenario"] = scenario_B
            df_combined = pd.concat(
                [df_A_copy, df_B_copy], ignore_index=True
            )

            st.markdown("### Export comparison data")
            csv_cmp = df_combined.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download comparison data (CSV)",
                data=csv_cmp,
                file_name="originforge_comparison_run.csv",
                mime="text/csv",
            )
        else:
            st.warning(
                "Comparison could not be completed; please check settings and try again."
            )

# ===================
# TAB 3: RUN HISTORY
# ===================
with tab_history:
    st.markdown(
        """
        #### 📜 Run History  
        A compact view of the most recent runs in this session.
        """.strip()
    )
    history = st.session_state["run_history"]
    if not history:
        st.info("No runs yet. Run a scenario in the **Single Scenario** tab to see history here.")
    else:
        st.dataframe(pd.DataFrame(history))

# ===================
# TAB 4: DEVELOPER VIEW
# ===================
with tab_dev:
    st.markdown(
        """
        #### 🧩 Developer View  
        Inspect and export the full JSON payload for the last single-scenario run.
        """.strip()
    )

    last_run = st.session_state.get("last_single_run", None)

    if last_run is None:
        st.info("No single scenario run in this session yet. Run one in the **Single Scenario** tab first.")
    else:
        df_last: pd.DataFrame = last_run["df"]
        preview_records = df_last.head(10).to_dict(orient="records")

        payload = {
            "scenario_name": last_run["name"],
            "params": last_run["params"],
            "ticks": last_run["ticks"],
            "metrics": df_last.to_dict(orient="records"),
        }

        st.markdown("##### JSON preview (first 10 records)")
        st.json(
            {
                "scenario_name": last_run["name"],
                "params": last_run["params"],
                "ticks": last_run["ticks"],
                "metrics_sample": preview_records,
            }
        )

        json_bytes = json.dumps(payload, indent=2).encode("utf-8")
        st.download_button(
            label="⬇️ Download full JSON payload",
            data=json_bytes,
            file_name="originforge_last_run.json",
            mime="application/json",
        )
