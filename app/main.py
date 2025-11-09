import json
from typing import Dict, Any, List

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from world import World
from utils import export_pdf_bytes

# ---------------------------
# Page Config & Global Setup
# ---------------------------
st.set_page_config(
    page_title="OriginForge — Policy Sandbox",
    page_icon="🌍",
    layout="wide",
)

# Initialize run history in session state
if "run_history" not in st.session_state:
    st.session_state["run_history"] = []  # list of dicts


# ---------------------------
# Constants & Scenario Config
# ---------------------------
SCENARIOS: Dict[str, Dict[str, Any] | None] = {
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
def run_world_with_params(params: Dict[str, Any], ticks: int) -> pd.DataFrame:
    """Helper to run a world and return its dataframe."""
    world = World(
        tax_rate=params["tax"],
        ubi_rate=params["ubi"],
        education_spend=params["edu"],
        resource_cap=params["cap"],
        regime=params["regime"],
    )
    df = world.run(ticks=int(ticks))
    return df


def get_params_for_scenario(name: str, sliders: Dict[str, Any]) -> Dict[str, Any]:
    """Return params dict given a scenario name and slider values."""
    if SCENARIOS[name] is None:
        # Custom mode: use sliders
        return {
            "tax": sliders["tax"],
            "ubi": sliders["ubi"],
            "edu": sliders["edu"],
            "cap": sliders["cap"],
            "regime": sliders["regime"],
        }
    else:
        return SCENARIOS[name]  # type: ignore


def pct_change(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return 100.0 * (new - old) / abs(old)


def describe_single_run(df: pd.DataFrame) -> str:
    """Create a simple human-readable summary for a single scenario run."""
    if df.empty:
        return "No data to summarize."

    first = df.iloc[0]
    last = df.iloc[-1]

    gdp_change = pct_change(last["gdp"], first["gdp"])
    gini_change = pct_change(last["gini_proxy"], first["gini_proxy"])
    stab_change = pct_change(last["stability"], first["stability"])

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
    """Convert run info + insight into a list of plain text lines for the PDF."""
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
        lines.append(f"- GDP (proxy): {last['gdp']:.2f}")
        lines.append(f"- Inequality (Gini proxy): {last['gini_proxy']:.3f}")
        lines.append(f"- Stability: {last['stability']:.3f}")
        lines.append(f"- Innovation: {last['innovation']:.3f}")
        lines.append(f"- Emissions: {last['emissions']:.3f}")
        lines.append("")
    else:
        lines.append("No metrics available (empty run).")
        lines.append("")

    lines.append("Quick Insight:")
    # Convert markdown bullets to plain text
    for line in insight_md.splitlines():
        plain = line.replace("• ", "- ").replace("**", "")
        lines.append(plain)

    return lines


def add_run_to_history(name: str, params: Dict[str, Any], df: pd.DataFrame) -> None:
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
            "Final GDP": round(last["gdp"], 1),
            "Final Gini": round(last["gini_proxy"], 3),
            "Final Stability": round(last["stability"], 3),
        },
    )
    # keep only last 10
    st.session_state["run_history"] = st.session_state["run_history"][:10]


# ---------------------------
# Layout: Header & About
# ---------------------------
header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        """
        ### 🌍 OriginForge — Policy Sandbox  
        Explore how tax, UBI, education, and resource policies shape a virtual society over time.
        """,
        unsafe_allow_html=False,
    )

with header_right:
    st.markdown(
        """
        <div style="text-align:right; font-size: 12px; color:#9CA3AF;">
        v1.0 • Simulation sandbox (educational & exploratory)
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("ℹ️ About & disclaimer", expanded=False):
    st.markdown(
        """
        **OriginForge** is a policy *sandbox* that models a simplified virtual economy.  
        It is designed for **education, experimentation, and scenario exploration** —  
        not as a precise forecast of any specific real-world country.
        """.strip()
    )


# ---------------------------
# Sidebar: Global Controls
# ---------------------------
with st.sidebar:
    st.markdown("## ⚙️ Scenario Controls")

    scenario_name_input = st.text_input(
        "Scenario name",
        value="My Scenario",
        help="This name will appear in the policy brief and run history.",
    )

    scenario_preset = st.selectbox(
        "Preset",
        list(SCENARIOS.keys()),
        help="Choose a preset, or 'Custom' to use the sliders below.",
        key="single_scenario",
    )

    st.markdown("**Policy sliders (used in Custom mode)**")

    tax = st.slider("Tax rate", 0.0, 0.50, 0.20, 0.01, key="single_tax")
    ubi = st.slider(
        "UBI (fraction of median income)", 0.0, 0.30, 0.10, 0.01, key="single_ubi"
    )
    edu = st.slider(
        "Education spend (GDP share)", 0.0, 0.10, 0.05, 0.01, key="single_edu"
    )
    cap = st.slider(
        "Resource cap (reduction)", 0.0, 0.40, 0.20, 0.01, key="single_cap"
    )
    regime = st.selectbox(
        "Regime", ["democracy", "autocracy"], key="single_regime"
    )

    ticks = st.number_input(
        "Simulation ticks",
        min_value=50,
        max_value=2000,
        value=200,
        step=50,
        key="single_ticks",
    )

    st.markdown("---")
    st.markdown("### 💾 Save / Load Config")

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
        "Load config (JSON)", type=["json"], key="config_uploader"
    )
    if uploaded is not None:
        try:
            loaded_cfg = json.loads(uploaded.read().decode("utf-8"))
            st.session_state["single_scenario"] = loaded_cfg.get(
                "scenario", scenario_preset
            )
            st.session_state["single_tax"] = loaded_cfg.get("tax", tax)
            st.session_state["single_ubi"] = loaded_cfg.get("ubi", ubi)
            st.session_state["single_edu"] = loaded_cfg.get("edu", edu)
            st.session_state["single_cap"] = loaded_cfg.get("cap", cap)
            st.session_state["single_regime"] = loaded_cfg.get("regime", regime)
            st.session_state["single_ticks"] = loaded_cfg.get("ticks", ticks)
            st.success("Config loaded. Sliders will reflect values on next rerun.")
        except Exception as e:
            st.error(f"Could not load config: {e}")


# ---------------------------
# Tabs: Single vs Compare
# ---------------------------
tab_single, tab_compare, tab_history = st.tabs(
    ["Single Scenario", "Compare Scenarios", "Run History"]
)

# ==============
# TAB 1: SINGLE
# ==============
with tab_single:
    sliders_single = {
        "tax": tax,
        "ubi": ubi,
        "edu": edu,
        "cap": cap,
        "regime": regime,
    }
    params_single = get_params_for_scenario(scenario_preset, sliders_single)

    col1, col2 = st.columns([2.2, 1.0])

    if st.button("▶️ Run single scenario", type="primary"):
        with st.spinner("Running simulation..."):
            try:
                df = run_world_with_params(params_single, ticks)
            except Exception as e:
                st.error(f"An error occurred while running the simulation: {e}")
                df = pd.DataFrame()

        if not df.empty:
            add_run_to_history(scenario_name_input, params_single, df)

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
                    height=450, margin=dict(l=10, r=10, t=30, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Advanced view: final distributions", expanded=False):
                    c1, c2 = st.columns(2)
                    # GDP vs Emissions scatter (final snapshot)
                    last = df.iloc[-1]
                    c1.metric("Final GDP", f"{last['gdp']:.1f}")
                    c1.metric("Final Emissions", f"{last['emissions']:.1f}")
                    c2.metric("Final Gini", f"{last['gini_proxy']:.3f}")
                    c2.metric("Final Stability", f"{last['stability']:.3f}")

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
                st.metric("GDP (proxy)", f'{last["gdp"]:.1f}')
                st.metric("Inequality (Gini)", f'{last["gini_proxy"]:.3f}')
                st.metric("Stability", f'{last["stability"]:.3f}')
                st.metric("Innovation", f'{last["innovation"]:.3f}')
                st.metric("Emissions", f'{last["emissions"]:.3f}')

                st.markdown("### Quick insight")
                insight_md = describe_single_run(df)
                st.markdown(insight_md)

                st.markdown("### Exports")
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
                    ticks=ticks,
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
    st.subheader("Compare two policy scenarios")

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
        max_value=2000,
        value=300,
        step=50,
        key="compare_ticks",
    )

    if st.button("▶️ Run comparison"):
        with st.spinner("Running comparison…"):
            try:
                params_A = SCENARIOS[scenario_A]  # type: ignore
                params_B = SCENARIOS[scenario_B]  # type: ignore

                df_A = run_world_with_params(params_A, ticks_cmp)
                df_B = run_world_with_params(params_B, ticks_cmp)
            except Exception as e:
                st.error(f"Error running comparison: {e}")
                df_A, df_B = pd.DataFrame(), pd.DataFrame()

        if not df_A.empty and not df_B.empty:
            # GDP comparison
            st.markdown("### GDP over time")
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
                height=380, margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_gdp, use_container_width=True)

            # Gini comparison
            st.markdown("### Inequality (Gini) over time")
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
                height=380, margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_gini, use_container_width=True)

            # Simple text insight
            last_A = df_A.iloc[-1]
            last_B = df_B.iloc[-1]

            gdp_diff = last_B["gdp"] - last_A["gdp"]
            gini_diff = last_B["gini_proxy"] - last_A["gini_proxy"]

            st.markdown("### Quick comparison insight")
            insight_lines: List[str] = []

            if gdp_diff > 0:
                insight_lines.append(
                    f"**{scenario_B}** ends with **higher GDP** than **{scenario_A}** by {gdp_diff:,.1f} units."
                )
            else:
                insight_lines.append(
                    f"**{scenario_A}** ends with **higher GDP** than **{scenario_B}** by {abs(gdp_diff):,.1f} units."
                )

            if gini_diff > 0:
                insight_lines.append(
                    f"**{scenario_B}** ends with **higher inequality (Gini)** than **{scenario_A}** by {gini_diff:.3f}."
                )
            else:
                insight_lines.append(
                    f"**{scenario_A}** ends with **higher inequality (Gini)** than **{scenario_B}** by {abs(gini_diff):.3f}."
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

            st.markdown("### Exports")
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
    st.subheader("Recent runs (this session)")
    history = st.session_state["run_history"]
    if not history:
        st.info("No runs yet. Run a scenario in the Single Scenario tab to see history here.")
    else:
        st.dataframe(pd.DataFrame(history))
