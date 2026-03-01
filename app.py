import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾")
st.title("🐾 Cat vs Dog Image Classifier")
st.write("Upload an image and the model will predict whether it's a **Cat** or a **Dog**.")

# ─── Load Model ─────────────────────────────────────────────────────────────
MODEL_PATH = "cat_dog_model.h5"  # Make sure this file is in your GitHub repo

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file '{MODEL_PATH}' not found. Please upload it to your GitHub repo.")
        st.stop()
    model = load_model(MODEL_PATH)
    return model

model = load_trained_model()
st.success("✅ Model loaded successfully!")

# ─── Image Upload ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img_display = Image.open(uploaded_file)
    st.image(img_display, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img_display.resize((64, 64))
    img = img.convert("RGB")           # ensure 3 channels
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Predicting..."):
        result = model.predict(img_array)
        confidence = float(result[0][0])

    # Show result
    st.subheader("🔍 Prediction Result")
    if confidence > 0.5:
        label = "🐶 Dog"
        prob = confidence
    else:
        label = "🐱 Cat"
        prob = 1 - confidence

    st.markdown(f"### {label}")
    st.progress(prob)
    st.write(f"Confidence: **{prob * 100:.2f}%**")

    # Bar chart
    fig, ax = plt.subplots()
    ax.bar(["Cat", "Dog"], [1 - confidence, confidence], color=["#FF6B6B", "#4ECDC4"])
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    ax.set_ylim(0, 1)
    st.pyplot(fig)
