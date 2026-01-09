import json
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
import os

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(
    page_title="Pomegranate Disease Detection",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎 Pomegranate Disease Detection")
st.write("Upload a pomegranate fruit image to predict the disease using Deep Learning.")

# -----------------------------
# Constants
# -----------------------------
IMAGE_SIZE = 256
MODEL_PATH = "pomegranate.keras"
CLASS_NAMES_PATH = "class_names.json"

# -----------------------------
# Load Model & Class Names
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found!")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        st.error("❌ class_names.json not found!")
        st.stop()
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)

model = load_model()
class_names = load_class_names()

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image):
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image)

    # Ensure RGB
    if image.shape[-1] != 3:
        image = image[:, :, :3]

    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# Prediction Function
# -----------------------------
def predict_disease(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    return class_names[class_index], confidence

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a pomegranate fruit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, width=300)

    if st.button("🔍 Predict Disease"):
        with st.spinner("🧠 Analyzing image..."):
            label, confidence = predict_disease(image)

        st.success(f"### 🍎 Prediction: **{label}**")
        st.info(f"### 📊 Confidence: **{confidence:.2f}%**")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<center>Built with ❤️ using TensorFlow & Streamlit</center>",
    unsafe_allow_html=True
)
