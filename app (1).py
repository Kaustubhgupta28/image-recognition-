import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PawSense AI",
    page_icon="🐾",
    layout="centered"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0d0d0d !important;
    color: #f0ece3 !important;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 20%, #1a1200 0%, #0d0d0d 60%) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 2rem 1.5rem 4rem !important; max-width: 720px !important; }

* { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; }

.hero { text-align: center; padding: 3rem 0 2rem; }
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #c9a84c, #f0d080);
    color: #0d0d0d;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 3.8rem !important;
    font-weight: 900 !important;
    line-height: 1.05 !important;
    color: #f0ece3 !important;
    margin: 0 0 0.5rem !important;
    letter-spacing: -1px;
}
.hero-title span {
    background: linear-gradient(135deg, #c9a84c, #f0d080);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub { color: #8a8070; font-size: 1rem; font-weight: 300; margin-top: 0.5rem; }
.gold-divider {
    width: 60px; height: 2px;
    background: linear-gradient(90deg, #c9a84c, #f0d080);
    margin: 1.5rem auto; border-radius: 2px;
}

[data-testid="stFileUploader"] {
    background: #161410 !important;
    border: 1.5px dashed #3a3020 !important;
    border-radius: 16px !important;
    padding: 1rem !important;
}
[data-testid="stImage"] img { border-radius: 16px !important; border: 1px solid #2a2010 !important; }

.result-card {
    background: linear-gradient(135deg, #161410 0%, #1c1810 100%);
    border: 1px solid #2a2010;
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin: 1.5rem 0;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #c9a84c, #f0d080, #c9a84c);
}
.result-animal {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem; font-weight: 900; margin: 0.3rem 0;
    background: linear-gradient(135deg, #c9a84c, #f0d080);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.result-emoji { font-size: 3.5rem; }
.result-label { color: #8a8070; font-size: 0.8rem; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.3rem; }
.confidence-row { display: flex; justify-content: space-between; margin-top: 1.5rem; gap: 1rem; }
.conf-block { flex: 1; background: #0d0d0d; border-radius: 12px; padding: 1rem; border: 1px solid #2a2010; }
.conf-block .animal { font-size: 1.4rem; margin-bottom: 0.3rem; }
.conf-block .pct { font-family: 'Playfair Display', serif; font-size: 1.6rem; font-weight: 700; color: #c9a84c; }
.conf-block .name { font-size: 0.75rem; color: #8a8070; letter-spacing: 0.1em; text-transform: uppercase; }
.bar-wrap { background: #1c1810; border-radius: 100px; height: 8px; margin: 1.2rem 0 0.4rem; overflow: hidden; border: 1px solid #2a2010; }
.bar-fill { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #c9a84c, #f0d080); }
.info-strip { display: flex; gap: 1rem; margin: 2rem 0 0; justify-content: center; }
.info-pill { background: #161410; border: 1px solid #2a2010; border-radius: 100px; padding: 0.4rem 1rem; font-size: 0.75rem; color: #8a8070; letter-spacing: 0.05em; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ AI Vision System</div>
    <div class="hero-title">Paw<span>Sense</span></div>
    <div class="gold-divider"></div>
    <div class="hero-sub">Upload a photo — our neural network identifies cats & dogs instantly</div>
</div>
""", unsafe_allow_html=True)

# ─── Load Model ──────────────────────────────────────────────────────────────
MODEL_PATH = "cat_dog_model.h5"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file 'cat_dog_model.h5' not found. Please upload it to your GitHub repo.")
        st.stop()
    return load_model(MODEL_PATH)

model = load_trained_model()

# ─── Upload ──────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop your image here or click to browse",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

if uploaded_file is not None:
    img_display = Image.open(uploaded_file).convert("RGB")
    st.image(img_display, use_column_width=True)

    img_resized = img_display.resize((64, 64))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing..."):
        result = model.predict(img_array, verbose=0)
        confidence = float(result[0][0])

    cat_pct = (1 - confidence) * 100
    dog_pct = confidence * 100

    if confidence > 0.5:
        label, emoji, main_pct = "Dog", "🐶", dog_pct
    else:
        label, emoji, main_pct = "Cat", "🐱", cat_pct

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Identified as</div>
        <div class="result-emoji">{emoji}</div>
        <div class="result-animal">{label}</div>
        <div class="bar-wrap">
            <div class="bar-fill" style="width:{main_pct:.1f}%"></div>
        </div>
        <div style="color:#8a8070; font-size:0.8rem; margin-bottom:1rem">{main_pct:.1f}% confidence</div>
        <div class="confidence-row">
            <div class="conf-block">
                <div class="animal">🐱</div>
                <div class="pct">{cat_pct:.1f}%</div>
                <div class="name">Cat</div>
            </div>
            <div class="conf-block">
                <div class="animal">🐶</div>
                <div class="pct">{dog_pct:.1f}%</div>
                <div class="name">Dog</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dark themed chart
    fig, ax = plt.subplots(figsize=(6, 2.5))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#161410')
    bars = ax.barh(
        ['Cat 🐱', 'Dog 🐶'],
        [cat_pct, dog_pct],
        color=['#c9a84c' if label == 'Cat' else '#2a2010',
               '#c9a84c' if label == 'Dog' else '#2a2010'],
        height=0.45, edgecolor='none'
    )
    ax.set_xlim(0, 110)
    ax.set_xlabel('Confidence %', color='#8a8070', fontsize=9)
    ax.tick_params(colors='#8a8070', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2010')
    for bar, val in zip(bars, [cat_pct, dog_pct]):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', color='#f0ece3', fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    <div class="info-strip">
        <div class="info-pill">📐 64×64 input</div>
        <div class="info-pill">🧠 CNN model</div>
        <div class="info-pill">⚡ Real-time</div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0; color: #3a3020; font-size: 0.85rem; letter-spacing:0.1em;">
        ↑ &nbsp; UPLOAD AN IMAGE TO BEGIN
    </div>
    """, unsafe_allow_html=True)
