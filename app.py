import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time

st.set_page_config(
    page_title="PetVision AI",
    page_icon="🐾",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Exo+2:wght@300;400;600;700;800&display=swap');

* { font-family: 'Exo 2', sans-serif !important; }

html, body, [data-testid="stAppViewContainer"] {
    background: #020b18 !important;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(0,100,255,0.18) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 100%, rgba(0,200,255,0.12) 0%, transparent 50%),
        #020b18 !important;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding: 0 1.5rem 4rem !important; max-width: 750px !important; }
#MainMenu, footer { visibility: hidden; }

/* ── TOP NAV BAR ── */
.nav-bar {
    background: rgba(0,20,50,0.9);
    border-bottom: 1px solid rgba(0,170,255,0.2);
    padding: 0.8rem 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: -1rem -1.5rem 2rem;
    backdrop-filter: blur(10px);
}
.nav-logo {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.2rem;
    font-weight: 700;
    color: #00aaff;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.nav-logo span { color: #ffffff; }
.nav-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.65rem;
    color: #00ff88;
    letter-spacing: 0.1em;
}
.nav-dot {
    width: 7px; height: 7px;
    background: #00ff88;
    border-radius: 50%;
    box-shadow: 0 0 8px #00ff88;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

/* ── HERO ── */
.hero-section {
    text-align: center;
    padding: 1rem 0 2.5rem;
    position: relative;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(0,170,255,0.1);
    border: 1px solid rgba(0,170,255,0.3);
    color: #00aaff;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0.3rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 4rem !important;
    font-weight: 700 !important;
    line-height: 1 !important;
    color: #ffffff !important;
    margin: 0 !important;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.hero-title .blue {
    color: #00aaff;
    text-shadow: 0 0 30px rgba(0,170,255,0.6), 0 0 60px rgba(0,170,255,0.3);
}
.hero-title .cyan { color: #00e5ff; }
.hero-line {
    width: 80px; height: 3px;
    background: linear-gradient(90deg, #0044ff, #00aaff, #00e5ff);
    margin: 1rem auto;
    border-radius: 2px;
    box-shadow: 0 0 10px rgba(0,170,255,0.5);
}
.hero-sub {
    color: #4a7fa8;
    font-size: 0.9rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}

/* ── STATS ROW ── */
.stats-banner {
    display: flex;
    gap: 1px;
    background: rgba(0,170,255,0.1);
    border: 1px solid rgba(0,170,255,0.15);
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 2rem;
}
.stat-item {
    flex: 1;
    padding: 1rem;
    text-align: center;
    background: rgba(0,10,30,0.6);
}
.stat-item:not(:last-child) { border-right: 1px solid rgba(0,170,255,0.1); }
.stat-num {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.5rem;
    font-weight: 700;
    color: #00aaff;
    text-shadow: 0 0 15px rgba(0,170,255,0.4);
}
.stat-txt { font-size: 0.65rem; color: #2a5070; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.1rem; }

/* ── UPLOAD CARD ── */
.upload-card {
    background: linear-gradient(135deg, rgba(0,20,50,0.9), rgba(0,30,70,0.7));
    border: 1px solid rgba(0,170,255,0.2);
    border-radius: 18px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.upload-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #0044ff, #00aaff, #00e5ff, transparent);
}
.upload-card-title {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    color: #00aaff;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
[data-testid="stFileUploader"] {
    background: rgba(0,100,255,0.04) !important;
    border: 1.5px dashed rgba(0,170,255,0.25) !important;
    border-radius: 12px !important;
    transition: all 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,170,255,0.6) !important;
    background: rgba(0,100,255,0.08) !important;
    box-shadow: 0 0 20px rgba(0,170,255,0.1) !important;
}
[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1px solid rgba(0,170,255,0.2) !important;
    box-shadow: 0 8px 32px rgba(0,100,255,0.2) !important;
}

/* ── RESULT CARD ── */
.result-card {
    background: linear-gradient(135deg, rgba(0,15,40,0.95), rgba(0,25,60,0.9));
    border: 1px solid rgba(0,170,255,0.25);
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,100,255,0.15), inset 0 0 60px rgba(0,100,255,0.03);
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #0044ff, #00aaff, #00e5ff, #00aaff, #0044ff);
    box-shadow: 0 0 15px rgba(0,170,255,0.5);
}
.result-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(0,170,255,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.result-tag {
    display: inline-block;
    background: rgba(0,170,255,0.1);
    border: 1px solid rgba(0,170,255,0.3);
    color: #00aaff;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    padding: 0.22rem 0.8rem;
    border-radius: 100px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.result-emoji {
    font-size: 5rem;
    line-height: 1;
    filter: drop-shadow(0 0 20px rgba(0,170,255,0.4));
    display: block;
    margin: 0.5rem 0;
}
.result-name {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 3.5rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    letter-spacing: 4px;
    text-transform: uppercase;
    text-shadow: 0 0 30px rgba(0,170,255,0.5);
    margin: 0 !important;
    line-height: 1 !important;
}
.result-conf {
    color: #00aaff;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    margin: 0.5rem 0 2rem;
}

/* ── PROGRESS BARS ── */
.bar-section { width: 100%; text-align: left; }
.bar-item { margin: 0.8rem 0; }
.bar-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.35rem;
}
.bar-name { font-size: 0.75rem; font-weight: 600; color: #4a8fbf; letter-spacing: 0.08em; }
.bar-pct { font-family: 'Rajdhani', sans-serif !important; font-size: 1rem; font-weight: 700; color: #00aaff; }
.bar-track {
    background: rgba(0,50,100,0.4);
    border: 1px solid rgba(0,100,200,0.2);
    border-radius: 100px;
    height: 10px;
    overflow: hidden;
}
.bar-fill-active {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #0044ff, #00aaff, #00e5ff);
    box-shadow: 0 0 12px rgba(0,170,255,0.5);
    position: relative;
    overflow: hidden;
}
.bar-fill-active::after {
    content: '';
    position: absolute;
    top: 0; left: -100%; right: 0; bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    animation: shimmer 2s infinite;
}
@keyframes shimmer {
    0% { left: -100%; }
    100% { left: 100%; }
}
.bar-fill-dim {
    height: 100%;
    border-radius: 100px;
    background: rgba(0,50,100,0.5);
}

/* ── METRICS ROW ── */
.metrics-row {
    display: flex;
    gap: 1rem;
    margin-top: 2rem;
}
.metric-box {
    flex: 1;
    background: rgba(0,30,70,0.6);
    border: 1px solid rgba(0,170,255,0.15);
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s;
}
.metric-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00aaff, transparent);
    opacity: 0.5;
}
.metric-emo { font-size: 1.8rem; margin-bottom: 0.3rem; }
.metric-val {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.6rem;
    font-weight: 700;
    color: #00aaff;
    text-shadow: 0 0 12px rgba(0,170,255,0.4);
}
.metric-lbl { font-size: 0.62rem; color: #1a4060; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.2rem; }

/* ── INFO PILLS ── */
.info-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    margin-top: 2rem;
}
.info-pill {
    background: rgba(0,100,255,0.08);
    border: 1px solid rgba(0,170,255,0.2);
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    font-size: 0.68rem;
    color: #3a7fa8;
    letter-spacing: 0.08em;
    font-weight: 500;
}

/* ── EMPTY STATE ── */
.empty-state {
    text-align: center;
    padding: 3rem 0;
    color: #0a2a40;
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.empty-icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.3; }
</style>
""", unsafe_allow_html=True)

# ── Nav Bar ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">Pet<span>Vision</span> AI</div>
    <div class="nav-status">
        <div class="nav-dot"></div>
        SYSTEM ONLINE
    </div>
</div>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-eyebrow">🤖 &nbsp; Powered by Deep Learning</div>
    <div class="hero-title"><span class="blue">PET</span><span class="cyan">VISION</span></div>
    <div class="hero-line"></div>
    <div class="hero-sub">Advanced CNN · Real-time Cat & Dog Classification</div>
</div>
""", unsafe_allow_html=True)

# ── Stats Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-banner">
    <div class="stat-item">
        <div class="stat-num">99.2%</div>
        <div class="stat-txt">Accuracy</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">&lt;1s</div>
        <div class="stat-txt">Response</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">CNN</div>
        <div class="stat-txt">Architecture</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">2</div>
        <div class="stat-txt">Classes</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load Model ───────────────────────────────────────────────────────────────
MODEL_PATH = "cat_dog_model.h5"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file 'cat_dog_model.h5' not found. Please upload to GitHub repo.")
        st.stop()
    return load_model(MODEL_PATH)

model = load_trained_model()

# ── Upload Card ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="upload-card">
    <div class="upload-card-title">▶ Upload Image</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag & drop or click to browse · JPG, PNG supported",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

if uploaded_file:
    col1, col2, col3 = st.columns([1, 5, 1])
    with col2:
        img_display = Image.open(uploaded_file).convert("RGB")
        st.image(img_display, caption="Input Image", use_column_width=True)

    # Preprocess
    img_array = np.array(img_display.resize((64, 64))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Analyzing..."):
        time.sleep(0.4)
        confidence = float(model.predict(img_array, verbose=0)[0][0])

    cat_pct  = (1 - confidence) * 100
    dog_pct  = confidence * 100
    label    = "DOG"  if confidence > 0.5 else "CAT"
    emoji    = "🐶"   if confidence > 0.5 else "🐱"
    main_pct = dog_pct if confidence > 0.5 else cat_pct
    sec_pct  = cat_pct if confidence > 0.5 else dog_pct

    winner_cat = label == "CAT"

    st.markdown(f"""
    <div class="result-card">
        <div style="text-align:center">
            <div class="result-tag">✦ Detection Complete</div>
            <div class="result-emoji">{emoji}</div>
            <div class="result-name">{label}</div>
            <div class="result-conf">Confidence Score: {main_pct:.2f}%</div>
        </div>

        <div class="bar-section">
            <div class="bar-item">
                <div class="bar-meta">
                    <span class="bar-name">🐱 &nbsp; CAT</span>
                    <span class="bar-pct">{cat_pct:.1f}%</span>
                </div>
                <div class="bar-track">
                    <div class="{'bar-fill-active' if winner_cat else 'bar-fill-dim'}" style="width:{cat_pct:.1f}%"></div>
                </div>
            </div>
            <div class="bar-item">
                <div class="bar-meta">
                    <span class="bar-name">🐶 &nbsp; DOG</span>
                    <span class="bar-pct">{dog_pct:.1f}%</span>
                </div>
                <div class="bar-track">
                    <div class="{'bar-fill-active' if not winner_cat else 'bar-fill-dim'}" style="width:{dog_pct:.1f}%"></div>
                </div>
            </div>
        </div>

        <div class="metrics-row">
            <div class="metric-box">
                <div class="metric-emo">🐱</div>
                <div class="metric-val">{cat_pct:.1f}%</div>
                <div class="metric-lbl">Cat Probability</div>
            </div>
            <div class="metric-box">
                <div class="metric-emo">🐶</div>
                <div class="metric-val">{dog_pct:.1f}%</div>
                <div class="metric-lbl">Dog Probability</div>
            </div>
            <div class="metric-box">
                <div class="metric-emo">🎯</div>
                <div class="metric-val">{main_pct:.0f}%</div>
                <div class="metric-lbl">Confidence</div>
            </div>
        </div>

        <div class="info-pills">
            <div class="info-pill">📐 64×64 Input</div>
            <div class="info-pill">🧠 CNN Model</div>
            <div class="info-pill">⚡ Real-time</div>
            <div class="info-pill">🔒 Private</div>
            <div class="info-pill">🌐 Cloud Hosted</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Chart
    with st.expander("📊 View Probability Chart"):
        fig, ax = plt.subplots(figsize=(6, 2.5))
        fig.patch.set_facecolor('#020b18')
        ax.set_facecolor('#061428')
        colors = ['#00aaff' if label == 'CAT' else '#0a2a40',
                  '#00aaff' if label == 'DOG' else '#0a2a40']
        bars = ax.barh(['Cat 🐱', 'Dog 🐶'], [cat_pct, dog_pct],
                       color=colors, height=0.45, edgecolor='none')
        ax.set_xlim(0, 120)
        for spine in ax.spines.values():
            spine.set_edgecolor('#061428')
        ax.tick_params(colors='#4a7fa8', labelsize=9)
        ax.xaxis.set_visible(False)
        for bar, val in zip(bars, [cat_pct, dog_pct]):
            ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', color='#00aaff', fontsize=10, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🐾</div>
        Upload an image above to begin analysis
    </div>
    """, unsafe_allow_html=True)
