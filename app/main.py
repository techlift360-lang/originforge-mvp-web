import streamlit as st
import plotly.graph_objects as go
from world import World
from utils import export_csv, export_pdf  # not used yet, but fine to keep

st.set_page_config(page_title="OriginForge Policy Sandbox", layout="wide")
st.title("🌍 OriginForge — Policy Sandbox (Web MVP)")

# Sidebar controls
with st.sidebar:
    st.header("Policy Controls")
    tax = st.slider("Tax Rate", 0.0, 0.50, 0.20, 0.01)
    ubi = st.slider("UBI (as fraction of median income)", 0.0, 0.30, 0.10, 0.01)
    edu = st.slider("Education Spend (GDP share)", 0.0, 0.10, 0.05, 0.01)
    cap = st.slider("Resource Cap (reduction)", 0.0, 0.40, 0.20, 0.01)
    regime = st.selectbox("Regime", ["democracy", "autocracy"])
    ticks = st.number_input("Simulation Ticks", min_value=50, max_value=2000, value=200, step=50)

col1, col2 = st.columns([2, 1])

if st.button("▶️ Run Simulation"):
    # Run the world
    world = World(
        tax_rate=tax,
        ubi_rate=ubi,
        education_spend=edu,
        resource_cap=cap,
        regime=regime,
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

    # --- NEW: CSV download button ---
    st.subheader("Exports")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download data as CSV",
        data=csv_data,
        file_name="originforge_run.csv",
        mime="text/csv",
    )
