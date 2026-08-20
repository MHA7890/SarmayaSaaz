import streamlit as st

st.set_page_config(
    page_title="FundForge AI Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(77,230,255,0.05), transparent 24%),
            radial-gradient(circle at 85% 0%, rgba(79,107,255,0.08), transparent 24%),
            #07101E;
        color: #EAF4FF;
        font-family: Inter, "Segoe UI", sans-serif;
    }
    .hero {
        text-align: center;
        padding: 4rem 0;
    }
    .hero h1 {
        font-size: 4rem;
        font-weight: 800;
        margin-bottom: 1rem;
        background: -webkit-linear-gradient(45deg, #4DE6FF, #435CFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        font-size: 1.5rem;
        color: #9FB4C8;
        max-width: 800px;
        margin: 0 auto 2rem auto;
    }
    .card {
        background: #111C2E;
        border: 1px solid #29425F;
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        border-color: #4DE6FF;
        box-shadow: 0 10px 30px rgba(77,230,255,0.1);
    }
    .card h3 {
        color: #F4FAFF;
        font-size: 1.8rem;
        margin-bottom: 1rem;
    }
    .card p {
        color: #9FB4C8;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>FundForge AI</h1>
    <p>The institutional-grade macro forecasting engine powered by Deep Learning, Multi-Source Sentiment Analysis, and Temporal Fusion Transformers.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <h3>📈 Commodities Engine</h3>
        <p>Analyze 6 global macro commodities including Gold, Crude Oil, and Copper using a robust 9-model ensemble trained on structural market metrics.</p>
        <p style="color: #4DE6FF; margin-top: 1rem;"><b>Select 'Commodities Engine' in the sidebar to launch.</b></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>🚀 Crypto Engine</h3>
        <p>Forecast 29 highly liquid crypto assets across 4 clusters, infused with 8.5 years of FinBERT NLP sentiment and Fear & Greed macro-regime context.</p>
        <p style="color: #4DE6FF; margin-top: 1rem;"><b>Select 'Crypto Engine' in the sidebar to launch.</b></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br><hr style='border-color: #29425F;'><center><p style='color: #9FB4C8;'>System Online • GPU: RTX 6000 Ada • DeepMind Antigravity Integration</p></center>", unsafe_allow_html=True)
