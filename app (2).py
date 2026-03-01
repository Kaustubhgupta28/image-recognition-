import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

st.set_page_config(
    page_title="PawSense AI 🐾",
    page_icon="🐾",
    layout="centered"
)

# ── Inject minimal CSS (only layout tweaks, no color overrides) ──────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&display=swap');

.big-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    text-align: center;
    margin-bottom: 0;
}
.gold { color: #c9a84c; }
.badge {
    text-align: center;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #c9a84c;
    margin-bottom: 0.5rem;
}
.divider {
    width: 50px; height: 2px;
    background: #c9a84c;
    margin: 0.8rem auto 1.5rem;
    border-radius: 2px;
}
.result-box {
    border: 1px solid #2a2010;
    border-radius: 18px;
    padding: 1.8rem;
    text-align: center;
    background: #1a1610;
    margin: 1rem 0;
    border-top: 3px solid #c9a84c;
}
.big-emoji { font-size: 4rem; }
.pred-label {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    font-weight: 900;
    color: #c9a84c;
}
.sub-label { font-size: 0.8rem; color: #8a8070; letter-spacing: 0.12em; text-transform: uppercase; }
.conf-grid { display: flex; gap: 1rem; margin-top: 1.2rem; }
.conf-cell {
    flex: 1; background: #0d0d0d;
    border-radius: 12px; padding: 1rem;
    border: 1px solid #2a2010; text-align: center;
}
.conf-cell .emo { font-size: 1.6rem; }
.conf-cell .pct { font-size: 1.5rem; font-weight: 700; color: #c9a84c; }
.conf-cell .nm  { font-size: 0.72rem; color: #8a8070; text-transform: uppercase; letter-spacing: 0.1em; }
.pills { display: flex; justify-content: center; gap: 0.8rem; margin-top: 1.5rem; flex-wrap: wrap; }
.pill {
    background: #1a1610; border: 1px solid #2a2010;
    border-radius: 100px; padding: 0.35rem 0.9rem;
    font-size: 0.73rem; color: #8a8070;
}
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="badge">✦ AI Vision System</div>', unsafe_allow_html=True)
st.markdown('<div class="big-title">Paw<span class="gold">Sense</span></div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#8a8070;font-size:0.95rem;">'
    'Upload a photo — our CNN identifies cats & dogs instantly</p>',
    unsafe_allow_html=True
)

# ── Load Model ───────────────────────────────────────────────────────────────
MODEL_PATH = "cat_dog_model.h5"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file 'cat_dog_model.h5' not found in the repo.")
        st.stop()
    return load_model(MODEL_PATH)

model = load_trained_model()

# ── Upload ───────────────────────────────────────────────────────────────────
st.markdown("---")
uploaded_file = st.file_uploader("📁 Choose an image (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        img_display = Image.open(uploaded_file).convert("RGB")
        st.image(img_display, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = np.array(img_display.resize((64, 64))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Analyzing..."):
        confidence = float(model.predict(img_array, verbose=0)[0][0])

    cat_pct = (1 - confidence) * 100
    dog_pct = confidence * 100
    label   = "Dog" if confidence > 0.5 else "Cat"
    emoji   = "🐶"  if confidence > 0.5 else "🐱"
    main_pct = dog_pct if confidence > 0.5 else cat_pct

    # ── Result card ──
    st.markdown(f"""
    <div class="result-box">
        <div class="sub-label">Identified as</div>
        <div class="big-emoji">{emoji}</div>
        <div class="pred-label">{label}</div>
        <div class="sub-label" style="margin-top:0.3rem">{main_pct:.1f}% confidence</div>
        <div class="conf-grid">
            <div class="conf-cell">
                <div class="emo">🐱</div>
                <div class="pct">{cat_pct:.1f}%</div>
                <div class="nm">Cat</div>
            </div>
            <div class="conf-cell">
                <div class="emo">🐶</div>
                <div class="pct">{dog_pct:.1f}%</div>
                <div class="nm">Dog</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Progress bars ──
    st.markdown("**Confidence Breakdown**")
    st.markdown("🐱 Cat")
    st.progress(cat_pct / 100)
    st.markdown("🐶 Dog")
    st.progress(dog_pct / 100)

    # ── Chart ──
    with st.expander("📊 View Probability Chart"):
        fig, ax = plt.subplots(figsize=(5, 2.2))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#1a1610')
        colors = ['#c9a84c' if label == 'Cat' else '#3a3020',
                  '#c9a84c' if label == 'Dog' else '#3a3020']
        bars = ax.barh(['Cat 🐱', 'Dog 🐶'], [cat_pct, dog_pct],
                       color=colors, height=0.4, edgecolor='none')
        ax.set_xlim(0, 115)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2010')
        ax.tick_params(colors='#c9a84c', labelsize=9)
        ax.xaxis.set_visible(False)
        for bar, val in zip(bars, [cat_pct, dog_pct]):
            ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', color='#f0ece3', fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Pills ──
    st.markdown("""
    <div class="pills">
        <div class="pill">📐 64×64 input</div>
        <div class="pill">🧠 CNN model</div>
        <div class="pill">⚡ Real-time inference</div>
        <div class="pill">🔒 No data stored</div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:2.5rem 0;color:#3a3020;
                font-size:0.85rem;letter-spacing:0.12em;">
        ↑ &nbsp; UPLOAD AN IMAGE TO BEGIN
    </div>
    """, unsafe_allow_html=True)
