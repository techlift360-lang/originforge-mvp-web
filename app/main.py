import streamlit as st
import plotly.graph_objects as go
from world import World
from utils import export_csv, export_pdf  # PDF not used yet, but kept for future

st.set_page_config(page_title="OriginForge Policy Sandbox", layout="wide")
st.title("🌍 OriginForge — Policy Sandbox (Web MVP)")

# --- Preset scenario definitions ---
SCENARIOS = {
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

# Sidebar controls
with st.sidebar:
    st.header("Policy Controls")

    scenario = st.selectbox(
        "Scenario",
        list(SCENARIOS.keys()),
        help="Choose a preset, or select 'Custom' to use the sliders below.",
    )

    st.markdown("**Manual Sliders (used only in Custom mode)**")

    tax = st.slider("Tax Rate", 0.0, 0.50, 0.20, 0.01)
    ubi = st.slider("UBI (as fraction of median income)", 0.0, 0.30, 0.10, 0.01)
    edu = st.slider("Education Spend (GDP share)", 0.0, 0.10, 0.05, 0.01)
    cap = st.slider("Resource Cap (reduction)", 0.0, 0.40, 0.20, 0.01)
    regime = st.selectbox("Regime", ["democracy", "autocracy"])
    ticks = st.number_input("Simulation Ticks", min_value=50, max_value=2000, value=200, step=50)

# Decide which parameters to actually use
if SCENARIOS[scenario] is None:
    # Custom mode: use sliders
    used_tax = tax
    used_ubi = ubi
    used_edu = edu
    used_cap = cap
    used_regime = regime
else:
    # Preset mode: override sliders
    preset = SCENARIOS[scenario]
    used_tax = preset["tax"]
    used_ubi = preset["ubi"]
    used_edu = preset["edu"]
    used_cap = preset["cap"]
    used_regime = preset["regime"]

col1, col2 = st.columns([2, 1])

if st.button("▶️ Run Simulation"):
    # Run the world with the chosen parameters
    world = World(
        tax_rate=used_tax,
        ubi_rate=used_ubi,
        education_spend=used_edu,
        resource_cap=used_cap,
        regime=used_regime,
    )
    df = world.run(ticks=int(ticks))

    # --- Left: charts ---
    with col1:
        st.subheader("Metrics Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["tick"], y=df["gdp"], name="GDP"))
        fig.add_trace(go.Scatter(x=df["tick"], y=df["gini_proxy"], name="Inequality (Gini proxy)"))
        fig.add_trace(go.Scatter(x=df["tick"], y=df["stability"], name="Stability"))
        fig.add_trace(go.Scatter(x=df["tick"], y=df["innovation"], name="Innovation"))
        fig.add_trace(go.Scatter(x=df["tick"], y=df["emissions"], name="Emissions"))
        fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Show which parameters were actually used
        st.markdown("### Parameters Used")
        st.write(
            {
                "Scenario": scenario,
                "Tax rate": round(used_tax, 3),
                "UBI (fraction)": round(used_ubi, 3),
                "Education (GDP share)": round(used_edu, 3),
                "Resource cap": round(used_cap, 3),
                "Regime": used_regime,
                "Ticks": int(ticks),
            }
        )

    # --- Right: summary metrics ---
    with col2:
        st.subheader("Summary")
        last = df.iloc[-1].to_dict()
        st.metric("Population", f'{int(last["population"])}')
        st.metric("GDP (proxy)", f'{last["gdp"]:.1f}')
        st.metric("Inequality (Gini)", f'{last["gini_proxy"]:.3f}')
        st.metric("Stability", f'{last["stability"]:.3f}')
        st.metric("Innovation", f'{last["innovation"]:.3f}')
        st.metric("Emissions", f'{last["emissions"]:.3f}')

    # --- CSV download ---
    st.subheader("Exports")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download data as CSV",
        data=csv_data,
        file_name="originforge_run.csv",
        mime="text/csv",
    )
