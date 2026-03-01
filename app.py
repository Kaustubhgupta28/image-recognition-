import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import time

st.set_page_config(
    page_title="PETSCAN AI",
    page_icon="🔮",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #050510 !important;
    color: #cc88ff !important;
}
[data-testid="stAppViewContainer"] {
    background:
        repeating-linear-gradient(0deg, transparent, transparent 40px, rgba(120,0,255,0.03) 40px, rgba(120,0,255,0.03) 41px),
        repeating-linear-gradient(90deg, transparent, transparent 40px, rgba(120,0,255,0.03) 40px, rgba(120,0,255,0.03) 41px),
        #050510 !important;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding: 0 1.5rem 4rem !important; max-width: 700px !important; }
* { font-family: 'Share Tech Mono', monospace !important; }
h1,h2,h3 { font-family: 'Orbitron', monospace !important; }
#MainMenu, footer { visibility: hidden; }

/* ── Scanline overlay ── */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,0.08) 3px, rgba(0,0,0,0.08) 4px);
    pointer-events: none;
    z-index: 9999;
}

/* ── Header ── */
.cyber-header {
    background: linear-gradient(135deg, #0d0025 0%, #140040 60%, #0a001a 100%);
    border: 1px solid #3a0a7a;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 1.5rem 0 1.8rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 40px rgba(120,0,255,0.2), inset 0 0 60px rgba(120,0,255,0.05);
}
.cyber-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #7700ff, #aa44ff, #ff44aa, #aa44ff, #7700ff, transparent);
    animation: scanline 3s linear infinite;
}
.cyber-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, #3a0a7a, transparent);
}
@keyframes scanline {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}
.cyber-grid {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(120,0,255,0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(120,0,255,0.06) 1px, transparent 1px);
    background-size: 30px 30px;
}
.cyber-badge {
    display: inline-block;
    border: 1px solid #7700ff;
    color: #aa44ff;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    padding: 0.28rem 0.9rem;
    border-radius: 3px;
    margin-bottom: 1rem;
    font-family: 'Orbitron', monospace !important;
    text-transform: uppercase;
    position: relative;
    box-shadow: 0 0 10px rgba(120,0,255,0.3), inset 0 0 10px rgba(120,0,255,0.05);
}
.cyber-title {
    font-family: 'Orbitron', monospace !important;
    font-size: 2.8rem !important;
    font-weight: 900 !important;
    color: #fff !important;
    text-shadow: 0 0 20px rgba(170,68,255,0.9), 0 0 50px rgba(120,0,255,0.5), 0 0 80px rgba(120,0,255,0.2);
    letter-spacing: 6px;
    position: relative;
    margin: 0 !important;
}
.cyber-title span { color: #aa44ff; }
.cyber-sub {
    color: #6633aa;
    font-size: 0.7rem;
    margin-top: 0.7rem;
    letter-spacing: 0.15em;
    position: relative;
}
.cyber-version {
    position: absolute;
    top: 0.8rem; right: 1rem;
    font-size: 0.6rem;
    color: #3a0a7a;
    letter-spacing: 0.1em;
}
.cyber-corner {
    position: absolute;
    width: 12px; height: 12px;
    border-color: #7700ff;
    border-style: solid;
}
.cyber-corner.tl { top: 8px; left: 8px; border-width: 1px 0 0 1px; }
.cyber-corner.tr { top: 8px; right: 8px; border-width: 1px 1px 0 0; }
.cyber-corner.bl { bottom: 8px; left: 8px; border-width: 0 0 1px 1px; }
.cyber-corner.br { bottom: 8px; right: 8px; border-width: 0 1px 1px 0; }

/* ── Upload Box ── */
.upload-panel {
    background: #0a0020;
    border: 1px solid #2a0a5a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    position: relative;
}
.upload-panel::before {
    content: '[ INPUT MODULE ]';
    position: absolute;
    top: -0.6rem; left: 1rem;
    background: #050510;
    padding: 0 0.5rem;
    font-size: 0.6rem;
    color: #7700ff;
    letter-spacing: 0.15em;
    font-family: 'Orbitron', monospace !important;
}
[data-testid="stFileUploader"] {
    background: rgba(120,0,255,0.04) !important;
    border: 1px dashed #3a0a7a !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #7700ff !important;
    box-shadow: 0 0 15px rgba(120,0,255,0.15) !important;
}
[data-testid="stImage"] img {
    border-radius: 10px !important;
    border: 1px solid #3a0a7a !important;
    box-shadow: 0 0 20px rgba(120,0,255,0.2) !important;
}

/* ── Result Panel ── */
.result-panel {
    background: linear-gradient(135deg, #0a0020, #140040 50%, #0a001a);
    border: 1px solid #3a0a7a;
    border-radius: 14px;
    padding: 2rem 1.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 40px rgba(120,0,255,0.15), inset 0 0 40px rgba(120,0,255,0.03);
    margin: 1.2rem 0;
}
.result-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #7700ff, #aa44ff, #7700ff, transparent);
}
.result-panel-label {
    position: absolute;
    top: -0.6rem; left: 50%; transform: translateX(-50%);
    background: #050510;
    padding: 0 0.8rem;
    font-size: 0.6rem;
    color: #aa44ff;
    letter-spacing: 0.2em;
    font-family: 'Orbitron', monospace !important;
    white-space: nowrap;
}
.result-emoji-cyber { font-size: 4rem; filter: drop-shadow(0 0 15px rgba(170,68,255,0.6)); }
.result-name-cyber {
    font-family: 'Orbitron', monospace !important;
    font-size: 2.2rem;
    font-weight: 900;
    color: #aa44ff;
    text-shadow: 0 0 20px rgba(170,68,255,0.8), 0 0 40px rgba(120,0,255,0.4);
    letter-spacing: 8px;
    margin: 0.5rem 0 0.3rem;
}
.result-conf-cyber {
    color: #6633aa;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    margin-bottom: 1.5rem;
}
.result-corners .rc { position: absolute; width: 10px; height: 10px; border-color: #aa44ff; border-style: solid; }
.rc.tl { top: 10px; left: 10px; border-width: 1px 0 0 1px; }
.rc.tr { top: 10px; right: 10px; border-width: 1px 1px 0 0; }
.rc.bl { bottom: 10px; left: 10px; border-width: 0 0 1px 1px; }
.rc.br { bottom: 10px; right: 10px; border-width: 0 1px 1px 0; }

/* ── Bars ── */
.cyber-bar-section { text-align: left; margin-top: 0.5rem; }
.cyber-bar-row { margin: 0.6rem 0; }
.cyber-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: #8844cc;
    letter-spacing: 0.1em;
    margin-bottom: 0.25rem;
    font-family: 'Orbitron', monospace !important;
}
.cyber-bar-track {
    background: #0d0025;
    border: 1px solid #2a0a5a;
    border-radius: 2px;
    height: 8px;
    overflow: hidden;
}
.cyber-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #5500cc, #7700ff, #aa44ff);
    border-radius: 2px;
    box-shadow: 0 0 8px rgba(170,68,255,0.6);
    position: relative;
}
.cyber-bar-fill::after {
    content: '';
    position: absolute;
    top: 0; right: 0; bottom: 0; width: 3px;
    background: #fff;
    opacity: 0.6;
    box-shadow: 0 0 6px #aa44ff;
}
.cyber-bar-fill.dim {
    background: linear-gradient(90deg, #1a0040, #2a0a5a);
    box-shadow: none;
}
.cyber-bar-fill.dim::after { display: none; }

/* ── Stats Grid ── */
.cyber-stats {
    display: flex;
    gap: 0.8rem;
    margin-top: 1.5rem;
}
.cyber-stat {
    flex: 1;
    background: #0a0020;
    border: 1px solid #2a0a5a;
    border-radius: 8px;
    padding: 0.9rem 0.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.cyber-stat::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, #7700ff, transparent);
}
.cyber-stat-emo { font-size: 1.4rem; }
.cyber-stat-val {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.2rem;
    font-weight: 700;
    color: #aa44ff;
    text-shadow: 0 0 10px rgba(170,68,255,0.5);
    margin: 0.2rem 0 0;
}
.cyber-stat-lbl { font-size: 0.58rem; color: #4a2a7a; letter-spacing: 0.12em; text-transform: uppercase; }

/* ── Pills ── */
.cyber-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
    margin-top: 1.5rem;
}
.cyber-pill {
    background: #0a0020;
    border: 1px solid #2a0a5a;
    border-radius: 3px;
    padding: 0.28rem 0.8rem;
    font-size: 0.62rem;
    color: #6633aa;
    letter-spacing: 0.1em;
}

/* ── Spinner override ── */
[data-testid="stSpinner"] p { color: #aa44ff !important; }

/* ── Empty state ── */
.cyber-empty {
    text-align: center;
    padding: 2rem 0;
    color: #2a0a5a;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cyber-header">
    <div class="cyber-grid"></div>
    <div class="cyber-corner tl"></div>
    <div class="cyber-corner tr"></div>
    <div class="cyber-corner bl"></div>
    <div class="cyber-corner br"></div>
    <div class="cyber-version">v2.0.1 // STABLE</div>
    <div class="cyber-badge">◈ &nbsp; NEURAL VISION SYSTEM</div>
    <div class="cyber-title">PET<span>SCAN</span></div>
    <div class="cyber-sub">// AI-POWERED PET CLASSIFICATION ENGINE //</div>
</div>
""", unsafe_allow_html=True)

# ── Load Model ───────────────────────────────────────────────────────────────
MODEL_PATH = "cat_dog_model.h5"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error("[ ERROR ] Model file 'cat_dog_model.h5' not found in repo.")
        st.stop()
    return load_model(MODEL_PATH)

model = load_trained_model()
st.markdown('<div style="text-align:center;font-size:0.65rem;color:#3a0a7a;letter-spacing:0.15em;margin-bottom:1rem;">[ SYSTEM ONLINE · MODEL LOADED · READY ]</div>', unsafe_allow_html=True)

# ── Upload ───────────────────────────────────────────────────────────────────
st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "UPLOAD IMAGE FILE [ JPG / PNG ]",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    col1, col2, col3 = st.columns([1, 5, 1])
    with col2:
        img_display = Image.open(uploaded_file).convert("RGB")
        st.image(img_display, caption="// INPUT FEED //", use_column_width=True)

    img_array = np.array(img_display.resize((64, 64))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("// SCANNING . ANALYZING . PROCESSING //"):
        time.sleep(0.5)
        confidence = float(model.predict(img_array, verbose=0)[0][0])

    cat_pct  = (1 - confidence) * 100
    dog_pct  = confidence * 100
    label    = "DOG"  if confidence > 0.5 else "CAT"
    emoji    = "🐶"   if confidence > 0.5 else "🐱"
    main_pct = dog_pct if confidence > 0.5 else cat_pct
    sec_pct  = cat_pct if confidence > 0.5 else dog_pct
    sec_lbl  = "CAT"  if confidence > 0.5 else "DOG"
    sec_emo  = "🐱"   if confidence > 0.5 else "🐶"

    st.markdown(f"""
    <div class="result-panel">
        <div class="result-panel-label">◈ DETECTION RESULT ◈</div>
        <div class="result-corners">
            <div class="rc tl"></div><div class="rc tr"></div>
            <div class="rc bl"></div><div class="rc br"></div>
        </div>

        <div class="result-emoji-cyber">{emoji}</div>
        <div class="result-name-cyber">{label}</div>
        <div class="result-conf-cyber">CONFIDENCE LEVEL :: {main_pct:.2f}%</div>

        <div class="cyber-bar-section">
            <div class="cyber-bar-row">
                <div class="cyber-bar-label"><span>🐱 &nbsp; CAT</span><span>{cat_pct:.1f}%</span></div>
                <div class="cyber-bar-track"><div class="cyber-bar-fill {'dim' if label=='DOG' else ''}" style="width:{cat_pct:.1f}%"></div></div>
            </div>
            <div class="cyber-bar-row">
                <div class="cyber-bar-label"><span>🐶 &nbsp; DOG</span><span>{dog_pct:.1f}%</span></div>
                <div class="cyber-bar-track"><div class="cyber-bar-fill {'dim' if label=='CAT' else ''}" style="width:{dog_pct:.1f}%"></div></div>
            </div>
        </div>

        <div class="cyber-stats">
            <div class="cyber-stat">
                <div class="cyber-stat-emo">🐱</div>
                <div class="cyber-stat-val">{cat_pct:.1f}%</div>
                <div class="cyber-stat-lbl">CAT PROB</div>
            </div>
            <div class="cyber-stat">
                <div class="cyber-stat-emo">🐶</div>
                <div class="cyber-stat-val">{dog_pct:.1f}%</div>
                <div class="cyber-stat-lbl">DOG PROB</div>
            </div>
            <div class="cyber-stat">
                <div class="cyber-stat-emo">🎯</div>
                <div class="cyber-stat-val">{main_pct:.0f}%</div>
                <div class="cyber-stat-lbl">ACCURACY</div>
            </div>
        </div>

        <div class="cyber-pills">
            <div class="cyber-pill">◈ 64×64 INPUT</div>
            <div class="cyber-pill">◈ CNN MODEL</div>
            <div class="cyber-pill">◈ REAL-TIME</div>
            <div class="cyber-pill">◈ ENCRYPTED</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Chart
    with st.expander("// VIEW PROBABILITY MATRIX //"):
        fig, ax = plt.subplots(figsize=(6, 2.5))
        fig.patch.set_facecolor('#050510')
        ax.set_facecolor('#0a0020')
        colors = ['#aa44ff' if label == 'CAT' else '#2a0a5a',
                  '#aa44ff' if label == 'DOG' else '#2a0a5a']
        bars = ax.barh(['CAT 🐱', 'DOG 🐶'], [cat_pct, dog_pct],
                       color=colors, height=0.4, edgecolor='#3a0a7a', linewidth=0.5)
        ax.set_xlim(0, 120)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a0a5a')
        ax.tick_params(colors='#6633aa', labelsize=8)
        ax.xaxis.set_visible(False)
        for bar, val in zip(bars, [cat_pct, dog_pct]):
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', color='#aa44ff', fontsize=9,
                    fontweight='bold', fontfamily='monospace')
        ax.set_facecolor('#0a0020')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

else:
    st.markdown("""
    <div class="cyber-empty">
        [ AWAITING INPUT ] &nbsp;·&nbsp; UPLOAD IMAGE TO INITIALIZE SCAN
    </div>
    """, unsafe_allow_html=True)
