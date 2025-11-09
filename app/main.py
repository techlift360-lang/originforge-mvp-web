"""
OriginForge — Policy Sandbox (Streamlit Front-End)

This module defines the full Streamlit app for:
- Overview / landing page
- Single-scenario simulations (with presets + custom sliders)
- Scenario comparison (A vs B)
- Run history (session)
- Developer view (JSON payload for the last single run)
- Distributions view (synthetic income & stability distributions)
- Sector view (industry / services / green GDP)

Backed by:
- world.World            -> simulation engine
- utils.export_pdf_bytes -> PDF policy brief generation
"""

from __future__ import annotations

import json
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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


def _get_query_params() -> Dict[str, str]:
    """
    Try to read query params in a version-agnostic way, and normalize values to simple strings.
    """
    qp: Dict[str, str] = {}
    try:
        # Newer Streamlit
        raw = st.query_params
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, list):
                    qp[k] = v[0]
                else:
                    qp[k] = str(v)
        else:
            qp = dict(raw)
    except Exception:
        # Older API
        try:
            raw = st.experimental_get_query_params()
            for k, v in raw.items():
                if isinstance(v, list) and v:
                    qp[k] = v[0]
                else:
                    qp[k] = str(v)
        except Exception:
            qp = {}
    return qp


def _apply_query_params_once() -> None:
    """
    Apply query params to session_state (only once) so you can share links like:

    ?preset=High%20UBI%20Safety%20Net&ticks=400&feedback=1&recession=1&climate=0&pop=800
    """
    if st.session_state.get("qp_applied", False):
        return

    qp = _get_query_params()
    if not qp:
        st.session_state["qp_applied"] = True
        return

    preset = qp.get("preset")
    if preset in SCENARIOS:
        st.session_state["single_scenario"] = preset

    def parse_int(val: Optional[str], lo: int, hi: int, default: int) -> int:
        if val is None:
            return default
        try:
            x = int(val)
        except ValueError:
            return default
        return max(lo, min(hi, x))

    def parse_bool(val: Optional[str], default: bool) -> bool:
        if val is None:
            return default
        v = val.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off"):
            return False
        return default

    ticks_from_qp = parse_int(qp.get("ticks"), 50, 5000, 200)
    pop_from_qp = parse_int(qp.get("pop"), 100, 5000, 500)

    st.session_state["single_ticks"] = ticks_from_qp
    st.session_state["population_size"] = pop_from_qp

    st.session_state["enable_recession"] = parse_bool(qp.get("recession"), False)
    st.session_state["enable_climate_shock"] = parse_bool(qp.get("climate"), False)
    st.session_state["enable_policy_feedback"] = parse_bool(qp.get("feedback"), False)

    st.session_state["qp_applied"] = True


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
    num_agents: int,
    recession: bool,
    climate_shock: bool,
    adapt_policies: bool,
) -> pd.DataFrame:
    """
    Cached simulation runner.

    Works with both:
    - New World(...) that accepts num_agents & adapt_policies
    - Older World(...) that may not (fallback without error)
    """
    ticks = max(1, min(int(ticks), 5000))
    num_agents = max(50, min(int(num_agents), 5000))

    # Try new signature first (with num_agents).
    # If the deployed World class is older, silently fall back.
    try:
        world = World(
            tax_rate=tax,
            ubi_rate=ubi,
            education_spend=edu,
            resource_cap=cap,
            regime=regime,
            num_agents=num_agents,
        )
    except TypeError:
        world = World(
            tax_rate=tax,
            ubi_rate=ubi,
            education_spend=edu,
            resource_cap=cap,
            regime=regime,
        )

    # Try to call with adapt_policies; fall back if not supported
    try:
        df = world.run(
            ticks=ticks,
            recession=recession,
            climate_shock=climate_shock,
            adapt_policies=adapt_policies,
        )
    except TypeError:
        df = world.run(
            ticks=ticks,
            recession=recession,
            climate_shock=climate_shock,
        )
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


def describe_single_run(
    df: pd.DataFrame,
    recession: bool,
    climate_shock: bool,
    adapt_policies: bool,
) -> str:
    """
    Create a human-readable story summary for a single scenario run.

    - Describes trends in GDP, inequality, and stability.
    - Labels the society as high/low growth, more/less equal, stable/fragile.
    - Mentions shocks and whether policies adapted over time.
    """
    if df.empty or len(df) < 2:
        return "Not enough data to summarize this run."

    first = df.iloc[0]
    last = df.iloc[-1]

    gdp_change = pct_change(float(last["gdp"]), float(first["gdp"]))
    gini_change = pct_change(float(last["gini_proxy"]), float(first["gini_proxy"]))
    stab_change = pct_change(float(last["stability"]), float(first["stability"]))

    lines: List[str] = []

    # GDP trend
    if gdp_change > 5:
        growth_label = "high-growth economy"
        lines.append(f"• GDP grew strongly by about **{gdp_change:.1f}%** over the simulation.")
    elif gdp_change > 1:
        growth_label = "moderate-growth economy"
        lines.append(f"• GDP increased by approximately **{gdp_change:.1f}%**.")
    elif gdp_change < -5:
        growth_label = "shrinking economy"
        lines.append(f"• GDP **fell sharply**, declining by about **{abs(gdp_change):.1f}%**.")
    elif gdp_change < -1:
        growth_label = "low-growth / shrinking economy"
        lines.append(f"• GDP decreased by about **{abs(gdp_change):.1f}%**.")
    else:
        growth_label = "flat-growth economy"
        lines.append("• GDP stayed relatively **flat** over the simulation.")

    # Inequality trend
    if gini_change < -3:
        equality_label = "more equal"
        lines.append(f"• Inequality (Gini) **decreased noticeably** by about {abs(gini_change):.1f}%.")
    elif gini_change < -1:
        equality_label = "slightly more equal"
        lines.append(f"• Inequality (Gini) decreased modestly by around {abs(gini_change):.1f}%.")
    elif gini_change > 3:
        equality_label = "more unequal"
        lines.append(f"• Inequality (Gini) **increased significantly** by about {gini_change:.1f}%.")
    elif gini_change > 1:
        equality_label = "slightly more unequal"
        lines.append(f"• Inequality (Gini) increased modestly by around {gini_change:.1f}%.")
    else:
        equality_label = "similar inequality"
        lines.append("• Inequality (Gini) remained **roughly constant**.")

    # Stability trend
    if stab_change > 5:
        stability_label = "more stable"
        lines.append(f"• Social stability **improved**, rising by about {stab_change:.1f}%.")
    elif stab_change > 1:
        stability_label = "slightly more stable"
        lines.append(f"• Social stability increased by roughly {stab_change:.1f}%.")
    elif stab_change < -5:
        stability_label = "fragile"
        lines.append(f"• Social stability **deteriorated**, falling by about {abs(stab_change):.1f}%.")
    elif stab_change < -1:
        stability_label = "slightly less stable"
        lines.append(f"• Social stability slipped by roughly {abs(stab_change):.1f}%.")
    else:
        stability_label = "similar stability"
        lines.append("• Social stability stayed **fairly stable**.")

    # Shocks context
    if recession and climate_shock:
        lines.append("• Both a **mid-run recession** and a **late climate shock** were applied.")
    elif recession:
        lines.append("• A **mid-run recession** was applied during the simulation.")
    elif climate_shock:
        lines.append("• A **late climate shock** was applied during the simulation.")
    else:
        lines.append("• No explicit shocks were applied; trends are driven purely by policy settings.")

    # Policy feedback context
    if adapt_policies:
        if "tax_rate" in df.columns and "ubi_rate" in df.columns:
            tax_diff = (float(df["tax_rate"].iloc[-1]) - float(df["tax_rate"].iloc[0])) * 100
            ubi_diff = (float(df["ubi_rate"].iloc[-1]) - float(df["ubi_rate"].iloc[0])) * 100
            edu_diff = (float(df["education_spend"].iloc[-1]) - float(df["education_spend"].iloc[0])) * 100
            cap_diff = (float(df["resource_cap"].iloc[-1]) - float(df["resource_cap"].iloc[0])) * 100

            changed = any(abs(x) > 0.2 for x in [tax_diff, ubi_diff, edu_diff, cap_diff])
            if changed:
                lines.append(
                    "• Government policies **adapted over time**, nudging tax, UBI, "
                    "education, and resource caps in response to conditions."
                )
            else:
                lines.append(
                    "• Policy feedback was enabled, but the government found little reason "
                    "to move far from the initial settings."
                )
        else:
            lines.append(
                "• Government policy feedback was enabled, guiding small adjustments to the levers over time."
            )
    else:
        lines.append("• Policy settings were held **fixed** throughout the run.")

    # Final one-line synthesis
    lines.append("")
    lines.append(
        f"_Overall, this run produces a **{growth_label}** that is **{equality_label}** with **{stability_label}** society._"
    )

    return "\n".join(lines)


def recommend_next_steps(
    df: pd.DataFrame,
    params: Dict[str, Any],
    recession: bool,
    climate_shock: bool,
    adapt_policies: bool,
) -> List[str]:
    """
    Rule-based "AI-like" recommendations for what to try next.
    This is deterministic and fully transparent (no external AI calls).
    """
    recs: List[str] = []
    if df.empty or len(df) < 2:
        recs.append("Run a scenario first to unlock targeted recommendations.")
        return recs

    last = df.iloc[-1]

    gdp = float(last["gdp"])
    gini = float(last["gini_proxy"])
    stab = float(last["stability"])
    emis = float(last["emissions"])
    inn = float(last["innovation"])

    # Inequality / stability
    if gini > 0.42 and stab < 0.6:
        recs.append(
            "To reduce inequality and improve stability, try **raising tax and UBI slightly** "
            "(e.g., +0.03 to tax and +0.05 to UBI) while keeping education stable."
        )
    elif gini < 0.30 and gdp < 1200:
        recs.append(
            "Inequality is already relatively low, but GDP is modest. "
            "You could **ease taxes slightly** and **boost education** to explore growth."
        )

    # Emissions
    if emis > 5.0:
        recs.append(
            "Emissions are high. Explore **tightening the resource cap** and/or increasing "
            "education to see if innovation-driven green growth can offset the drag."
        )
    elif emis < 1.5 and gdp is not None and gdp < 1500:
        recs.append(
            "Emissions are low but GDP is modest. It may be safe to **loosen the resource cap** a bit "
            "to see if you can gain growth without losing too much on the climate side."
        )

    # Innovation
    if inn < 0.25:
        recs.append(
            "Innovation is relatively low. Consider **raising education spend** and turning on "
            "**policy feedback** so the system can adjust over time."
        )
    elif inn > 0.6 and emis > 3.0:
        recs.append(
            "Innovation is high but emissions are elevated. Try combining this with **stronger caps** "
            "to see if green sector growth can keep GDP stable."
        )

    # Shocks
    if not recession or not climate_shock:
        recs.append(
            "Once you like a baseline, **add a recession or climate shock** to test the resilience "
            "of your policy mix."
        )

    # Policy feedback
    if not adapt_policies:
        recs.append(
            "Enable **Adaptive government (policy feedback)** to see how an algorithmic policymaker "
            "would gradually adjust tax, UBI, education, and caps."
        )

    if not recs:
        recs.append(
            "This configuration looks balanced. Next, try **changing one lever at a time** "
            "(e.g., +0.05 tax or +0.03 UBI) and compare paths in the **Compare Scenarios** tab."
        )

    return recs


def build_policy_brief_lines(
    scenario_name: str,
    params: Dict[str, Any],
    ticks: int,
    df: pd.DataFrame,
    insight_md: str,
    num_agents: int,
) -> List[str]:
    """
    Convert run info + insight into a list of plain-text lines for PDF export.
    """
    lines: List[str] = []
    lines.append(f"Scenario: {scenario_name}")
    lines.append(f"Ticks simulated: {int(ticks)}")
    lines.append(f"Population (agents): {int(num_agents)}")
    lines.append("")
    lines.append("Policy Parameters (initial):")
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
        if "gdp_industry" in df.columns:
            lines.append(
                f"- Sector GDP (industry / services / green): "
                f"{float(last['gdp_industry']):.1f} / {float(last['gdp_services']):.1f} / {float(last['gdp_green']):.1f}"
            )
        if "tax_rate" in df.columns:
            lines.append("")
            lines.append("Final Policy Levels (after feedback):")
            lines.append(f"- Tax rate: {float(last['tax_rate']):.3f}")
            lines.append(f"- UBI rate: {float(last['ubi_rate']):.3f}")
            lines.append(f"- Education spend: {float(last['education_spend']):.3f}")
            lines.append(f"- Resource cap: {float(last['resource_cap']):.3f}")
        lines.append("")
    else:
        lines.append("No metrics available (empty run).")
        lines.append("")

    lines.append("Quick Insight:")
    for line in insight_md.splitlines():
        plain = line.replace("• ", "- ").replace("**", "")
        lines.append(plain)

    return lines


def add_run_to_history(
    name: str,
    params: Dict[str, Any],
    df: pd.DataFrame,
    num_agents: int,
) -> None:
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
            "Population": int(num_agents),
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
    st.session_state["run_history"] = st.session_state["run_history"][:10]


# Apply query params only once per session (for shareable URLs)
_apply_query_params_once()

# ---------------------------
# Header & Intro Section
# ---------------------------

# Try to show the same banner as in the README, if available
try:
    st.image("assets/originforge-banner.png", width=800)
except Exception:
    pass

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
        v1.3 • Simulation sandbox (educational & exploratory)
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

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
        - The world contains a stylized population, firms, and a government.  
        - Policies influence income, redistribution, innovation, emissions, stability, and sectors.  
        - Each **tick** represents a time step where the system evolves.  
        - Metrics are proxies, calibrated for exploration rather than precise forecasts.
        """.strip()
    )

with st.expander("About & disclaimer", expanded=False):
    st.markdown(
        """
        **OriginForge** is a policy *sandbox* for educational and exploratory use.  
        It is *not* a calibrated forecast model for any specific country or real-world policy decision.
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
        value=st.session_state.get("single_ticks", 200),
        step=50,
        help="Higher values simulate longer time horizons.",
        key="single_ticks",
    )

    population_size = st.number_input(
        "Population size (agents)",
        min_value=100,
        max_value=5000,
        value=st.session_state.get("population_size", 500),
        step=100,
        help="Larger populations are more realistic but slower to simulate.",
        key="population_size",
    )

    st.markdown("#### Shocks (optional)")

    enable_recession = st.checkbox(
        "Mid-run economic recession",
        value=st.session_state.get("enable_recession", False),
        help="At halfway through the simulation, GDP and stability take a temporary hit.",
        key="enable_recession",
    )

    enable_climate_shock = st.checkbox(
        "Late climate shock",
        value=st.session_state.get("enable_climate_shock", False),
        help="Near the end of the run, emissions spike and stability suffers.",
        key="enable_climate_shock",
    )

    st.markdown("#### Policy feedback")

    enable_policy_feedback = st.checkbox(
        "Adaptive government (policy feedback)",
        value=st.session_state.get("enable_policy_feedback", False),
        help="Allow the government to nudge tax, UBI, education, and resource caps "
             "in response to inequality, growth, emissions, and stability.",
        key="enable_policy_feedback",
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
        "population_size": int(population_size),
        "enable_recession": bool(enable_recession),
        "enable_climate_shock": bool(enable_climate_shock),
        "enable_policy_feedback": bool(enable_policy_feedback),
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
            st.session_state["population_size"] = clamp_int(
                loaded_cfg.get("population_size", population_size),
                100,
                5000,
                int(population_size),
            )
            st.session_state["enable_recession"] = bool(
                loaded_cfg.get("enable_recession", enable_recession)
            )
            st.session_state["enable_climate_shock"] = bool(
                loaded_cfg.get("enable_climate_shock", enable_climate_shock)
            )
            st.session_state["enable_policy_feedback"] = bool(
                loaded_cfg.get("enable_policy_feedback", enable_policy_feedback)
            )
            st.success("Config loaded. Controls will update on next rerun.")
        except Exception as e:
            st.error(f"Could not load config: {e}")

# ---------------------------
# Tabs: Main Sections
# ---------------------------

tab_overview, tab_single, tab_compare, tab_history, tab_dev, tab_dist, tab_sectors = st.tabs(
    [
        "Overview",
        "Single Scenario",
        "Compare Scenarios",
        "Run History",
        "Developer View",
        "Distributions",
        "Sectors",
    ]
)

# ==============
# TAB 0: OVERVIEW
# ==============
with tab_overview:
    st.markdown(
        """
        ### 🧭 What is OriginForge?

        OriginForge is an **AI-inspired civilization sandbox** for exploring how policy levers like
        tax, UBI, education, and resource caps shape a virtual society over time.

        It is designed for:

        - 🎓 **Policy schools & grad programs** — teaching trade-offs and dynamic effects  
        - 🧠 **Think tanks & NGOs** — stress-testing ideas before building full models  
        - 🏢 **Strategy & foresight teams** — running "what if" experiments quickly  

        ---
        ### 🧪 Example questions you can explore

        - What happens if we **raise UBI** while keeping taxes moderate?  
        - Can we **cut emissions** with strong caps and still keep GDP growing?  
        - How do **education investments** change innovation and stability over time?  
        - What policy mix is **more resilient** to a recession or climate shock?  

        ---
        ### 🚀 How to use this app

        1. Go to **Single Scenario** and either pick a preset or use **Custom (use sliders)**.  
        2. Choose **ticks** (time horizon) and **population size** in the left sidebar.  
        3. Optionally enable **recession / climate shocks** and **policy feedback**.  
        4. Click **Run single scenario** to generate metrics, a narrative, and exports.  
        5. Use **Compare Scenarios** to contrast different policy regimes.  
        6. Explore **Distributions** and **Sectors** to see structural effects.  

        This sandbox is **exploratory**, not predictive — it is meant to help conversation,
        hypothesis-building, and teaching.
        """.strip()
    )

# ==============
# TAB 1: SINGLE
# ==============
with tab_single:
    st.markdown(
        """
        #### 🎯 Single Scenario  
        Configure one policy setup and see how the virtual society evolves over time.
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
                    int(population_size),
                    bool(enable_recession),
                    bool(enable_climate_shock),
                    bool(enable_policy_feedback),
                )
            except Exception as e:
                st.error(f"An error occurred while running the simulation: {e}")
                df = pd.DataFrame()

        if not df.empty:
            add_run_to_history(
                scenario_name_input,
                params_single,
                df,
                int(population_size),
            )
            st.session_state["last_single_run"] = {
                "name": scenario_name_input,
                "params": params_single,
                "ticks": int(ticks),
                "df": df,
                "population_size": int(population_size),
                "enable_recession": bool(enable_recession),
                "enable_climate_shock": bool(enable_climate_shock),
                "enable_policy_feedback": bool(enable_policy_feedback),
            }

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

                if enable_recession:
                    fig.add_vline(
                        x=int(ticks) // 2,
                        line_dash="dash",
                        line_color="orange",
                        annotation_text="Recession",
                        annotation_position="top left",
                    )
                if enable_climate_shock:
                    fig.add_vline(
                        x=int(int(ticks) * 0.8),
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Climate shock",
                        annotation_position="top right",
                    )

                fig.update_layout(
                    height=450,
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Advanced view: final snapshot", expanded=False):
                    c1_adv, c2_adv = st.columns(2)
                    last_row = df.iloc[-1]
                    c1_adv.metric("Final GDP", f"{float(last_row['gdp']):.1f}")
                    c1_adv.metric("Final Emissions", f"{float(last_row['emissions']):.1f}")
                    c2_adv.metric("Final Gini", f"{float(last_row['gini_proxy']):.3f}")
                    c2_adv.metric("Final Stability", f"{float(last_row['stability']):.3f}")

                    if "tax_rate" in df.columns:
                        st.markdown("**Policy path (first → last):**")
                        st.write(
                            {
                                "Tax": f"{df['tax_rate'].iloc[0]:.3f} → {df['tax_rate'].iloc[-1]:.3f}",
                                "UBI": f"{df['ubi_rate'].iloc[0]:.3f} → {df['ubi_rate'].iloc[-1]:.3f}",
                                "Education": f"{df['education_spend'].iloc[0]:.3f} → {df['education_spend'].iloc[-1]:.3f}",
                                "Cap": f"{df['resource_cap'].iloc[0]:.3f} → {df['resource_cap'].iloc[-1]:.3f}",
                            }
                        )

                st.markdown("### Parameters used")
                st.write(
                    {
                        "Scenario preset": scenario_preset,
                        "Scenario name": scenario_name_input,
                        "Population": int(population_size),
                        "Tax rate (initial)": round(params_single["tax"], 3),
                        "UBI (fraction, initial)": round(params_single["ubi"], 3),
                        "Education (GDP share, initial)": round(params_single["edu"], 3),
                        "Resource cap (initial)": round(params_single["cap"], 3),
                        "Regime": params_single["regime"],
                        "Ticks": int(ticks),
                        "Recession": bool(enable_recession),
                        "Climate shock": bool(enable_climate_shock),
                        "Policy feedback": bool(enable_policy_feedback),
                    }
                )

            with col2:
                st.subheader("Summary snapshot")
                last = df.iloc[-1].to_dict()
                st.metric("Population", f'{int(last["population"])}')
                st.metric("GDP (proxy)", f'{float(last["gdp"]):.1f}')
                st.metric("Inequality (Gini)", f'{float(last["gini_proxy"]):.3f}')
                st.metric("Stability", f'{float(last["stability"]):.3f}')
                st.metric("Innovation", f'{float(last["innovation"]):.3f}')
                st.metric("Emissions", f'{float(last["emissions"]):.3f}')

                st.markdown("### Simulation story")
                insight_md = describe_single_run(
                    df,
                    recession=bool(enable_recession),
                    climate_shock=bool(enable_climate_shock),
                    adapt_policies=bool(enable_policy_feedback),
                )
                st.markdown(insight_md)

                st.markdown("### Suggested next experiments")
                recs = recommend_next_steps(
                    df,
                    params_single,
                    recession=bool(enable_recession),
                    climate_shock=bool(enable_climate_shock),
                    adapt_policies=bool(enable_policy_feedback),
                )
                for r in recs:
                    st.markdown(f"- {r}")

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
                    num_agents=int(population_size),
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

    population_cmp = st.number_input(
        "Population size (agents, both scenarios)",
        min_value=100,
        max_value=5000,
        value=500,
        step=100,
        key="compare_population",
    )

    run_compare = st.button("▶️ Run comparison")
    st.markdown("")

    if run_compare:
        with st.spinner("Running comparison…"):
            try:
                params_A = SCENARIOS[scenario_A]  # type: ignore[assignment]
                params_B = SCENARIOS[scenario_B]  # type: ignore[assignment]

                # For fair comparison, keep policy feedback + shocks off here
                df_A = run_world_with_params_cached(
                    params_A["tax"],
                    params_A["ubi"],
                    params_A["edu"],
                    params_A["cap"],
                    params_A["regime"],
                    int(ticks_cmp),
                    int(population_cmp),
                    False,
                    False,
                    False,
                )
                df_B = run_world_with_params_cached(
                    params_B["tax"],
                    params_B["ubi"],
                    params_B["edu"],
                    params_B["cap"],
                    params_B["regime"],
                    int(ticks_cmp),
                    int(population_cmp),
                    False,
                    False,
                    False,
                )
            except Exception as e:
                st.error(f"Error running comparison: {e}")
                df_A, df_B = pd.DataFrame(), pd.DataFrame()

        if not df_A.empty and not df_B.empty:
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
            "population_size": last_run.get("population_size", 500),
            "shocks": {
                "recession": last_run.get("enable_recession", False),
                "climate_shock": last_run.get("enable_climate_shock", False),
            },
            "policy_feedback": last_run.get("enable_policy_feedback", False),
            "metrics": df_last.to_dict(orient="records"),
        }

        st.markdown("##### JSON preview (first 10 records)")
        st.json(
            {
                "scenario_name": last_run["name"],
                "params": last_run["params"],
                "ticks": last_run["ticks"],
                "population_size": last_run.get("population_size", 500),
                "policy_feedback": last_run.get("enable_policy_feedback", False),
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

# ===================
# TAB 5: DISTRIBUTIONS
# ===================
with tab_dist:
    st.markdown(
        """
        #### 📦 Distributions  
        Visualize synthetic income and stability distributions for the last single scenario.
        """.strip()
    )

    last_run = st.session_state.get("last_single_run", None)
    if last_run is None:
        st.info("Run a single scenario first to see distributions.")
    else:
        df_last: pd.DataFrame = last_run["df"]
        last_row = df_last.iloc[-1]

        np.random.seed(42)
        pop_size = int(last_run.get("population_size", 500))

        incomes = np.random.lognormal(
            mean=3.0,
            sigma=0.4,
            size=pop_size,
        ) * max(float(last_row["gdp"]) / 5000.0, 0.1)

        stability_vals = np.random.normal(
            loc=float(last_row["stability"]),
            scale=0.05,
            size=pop_size,
        )

        col_income, col_stab = st.columns(2)

        with col_income:
            st.subheader("Income distribution (synthetic)")
            st.bar_chart(pd.Series(incomes, name="income"))

        with col_stab:
            st.subheader("Stability distribution (synthetic)")
            st.bar_chart(pd.Series(stability_vals, name="stability"))

# ===================
# TAB 6: SECTORS
# ===================
with tab_sectors:
    st.markdown(
        """
        #### 🏗 Sector View  
        See how industry, services, and green sectors contribute to total GDP over time.
        """.strip()
    )

    last_run = st.session_state.get("last_single_run", None)
    if last_run is None:
        st.info("Run a single scenario first to see sector trends.")
    else:
        df_last: pd.DataFrame = last_run["df"]

        required_cols = {"gdp_industry", "gdp_services", "gdp_green"}
        if not required_cols.issubset(df_last.columns):
            st.warning(
                "Sector data is not available for this run. "
                "Try rerunning a scenario to regenerate metrics."
            )
        else:
            fig_sect = go.Figure()
            fig_sect.add_trace(
                go.Scatter(
                    x=df_last["tick"],
                    y=df_last["gdp_industry"],
                    stackgroup="one",
                    name="Industry",
                )
            )
            fig_sect.add_trace(
                go.Scatter(
                    x=df_last["tick"],
                    y=df_last["gdp_services"],
                    stackgroup="one",
                    name="Services",
                )
            )
            fig_sect.add_trace(
                go.Scatter(
                    x=df_last["tick"],
                    y=df_last["gdp_green"],
                    stackgroup="one",
                    name="Green",
                )
            )
            fig_sect.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                yaxis_title="Sector GDP (proxy)",
            )
            st.plotly_chart(fig_sect, use_container_width=True)

            with st.expander("Final sector snapshot", expanded=False):
                last_row = df_last.iloc[-1]
                total_gdp = float(last_row["gdp"]) if "gdp" in df_last.columns else None
                if total_gdp and total_gdp > 0:
                    st.write(
                        {
                            "Industry share": f"{100.0 * float(last_row['gdp_industry']) / total_gdp:.1f}%",
                            "Services share": f"{100.0 * float(last_row['gdp_services']) / total_gdp:.1f}%",
                            "Green share": f"{100.0 * float(last_row['gdp_green']) / total_gdp:.1f}%",
                        }
                    )
                else:
                    st.write(
                        {
                            "Industry": float(last_row["gdp_industry"]),
                            "Services": float(last_row["gdp_services"]),
                            "Green": float(last_row["gdp_green"]),
                        }
                    )
