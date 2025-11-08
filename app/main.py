import streamlit as st
import plotly.graph_objects as go
from app.world import World
from app.utils import export_csv, export_pdf

import streamlit as st
import plotly.graph_objects as go
from app.world import World
from app.utils import export_csv, export_pdf

st.set_page_config(page_title="OriginForge Policy Sandbox", layout="wide")
st.title("🌍 OriginForge — Policy Sandbox (Web MVP)")

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
    world = World(tax_rate=tax, ubi_rate=ubi, education_spend=edu, resource_cap=cap, regime=regime)
    df = world.run(ticks=int(ticks))

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

    with col2:
        st.subheader("Summary")
        last = df.iloc[-1].to_dict()
        st.metric("Population", f'{int(last["population"])}')
        st.metric("GDP (proxy)", f'{last["gdp"]:.1f}')
        st.metric("Inequality (Gini)", f'{last["gini_proxy"]:.3f}')
        st.metric("Stability", f'{last["stability"]:.3f}')
        import streamlit as st
import plotly.graph_objects as go
from world import World
from utils import export_csv, export_pdf

