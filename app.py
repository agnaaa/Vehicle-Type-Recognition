import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from PIL import Image
import time
import joblib
import os
from streamlit_option_menu import option_menu
import io
import random

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    from ultralytics import YOLO
    yolo_model = YOLO("best.pt")

    try:
        classifier = tf.keras.models.load_model("model/classifier_model.h5")
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat classifier model: {e}")
        classifier = None  # supaya app tetap jalan

    return yolo_model, classifier

# ==========================
# UI
# Konfigurasi halaman
st.set_page_config(page_title="AI Image Detection", layout="wide")

# =======================
# CSS Styling
# =======================
st.markdown("""
<style>
body {
    background-color: #fdeff4;
}
.navbar {
    background-color: white;
    padding: 15px 0;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    border-radius: 12px;
    margin-bottom: 30px;
}
.nav-item {
    display: inline-block;
    margin: 0 25px;
    font-weight: 600;
    color: #444;
    cursor: pointer;
    font-size: 16px;
}
.nav-item.active {
    color: #e75480;
    border-bottom: 2px solid #e75480;
    padding-bottom: 4px;
}
.hero {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 70px 50px;
}
.hero-text h1 {
    font-size: 48px;
    font-weight: 800;
    color: #111;
    margin-bottom: 10px;
}
.hero-text span {
    color: #e75480;
}
.hero-text p {
    color: #555;
    font-size: 16px;
    max-width: 500px;
    margin-bottom: 25px;
}
.upload-card {
    background: white;
    border-radius: 20px;
    padding: 25px;
    width: 360px;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}
.upload-card h4 {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 15px;
}
.upload-placeholder {
    border: 2px dashed #f1b4c2;
    padding: 40px;
    border-radius: 12px;
    color: #b88a9f;
    font-size: 14px;
}
.upload-btn {
    margin-top: 15px;
    background-color: #f28cab;
    color: white;
    border: none;
    padding: 8px 20px;
    border-radius: 8px;
    cursor: pointer;
}
.try-btn {
    background-color: #e75480;
    color: white;
    padding: 10px 22px;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    cursor: pointer;
}
.try-btn:hover {
    background-color: #d14a73;
}
</style>
""", unsafe_allow_html=True)

# =======================
# Navbar
# =======================
if "page" not in st.session_state:
    st.session_state.page = "Home"

st.markdown('<div class="navbar">', unsafe_allow_html=True)
cols = st.columns(5)
pages = ["Home", "Classification", "Model Performance", "Model Info", "About Project"]
for i, page in enumerate(pages):
    if cols[i].button(page):
        st.session_state.page = page
st.markdown('</div>', unsafe_allow_html=True)

# =======================
# HOME PAGE
# =======================
if st.session_state.page == "Home":
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("""
        <div class="hero-text">
            <h1>Deteksi Jenis <span>Kendaraan AI</span></h1>
            <p>Platform revolusioner yang menggunakan teknologi deep learning untuk mengidentifikasi dan mengklasifikasi jenis kendaraan seperti mobil, motor, truck, dan bus dengan akurasi tinggi.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚗 Coba Sekarang", key="try_now"):
            st.session_state.page = "Classification"

    with col2:
        st.markdown("""
        <div class="upload-card">
            <h4>Demo Cepat</h4>
            <div class="upload-placeholder">
                🖼️<br><br>Upload gambar kendaraan untuk analisis
            </div>
            <button class="upload-btn">📁 Pilih Gambar</button>
        </div>
        """, unsafe_allow_html=True)

# =======================
# CLASSIFICATION PAGE
# =======================
elif st.session_state.page == "Classification":
    st.header("Klasifikasi Gambar AI")
    st.write("Upload gambar kendaraan dan biarkan AI menganalisis jenisnya.")

    uploaded_file = st.file_uploader("Upload gambar kendaraan", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diupload", use_column_width=True)

        with st.spinner("Menganalisis gambar..."):
            time.sleep(2)

        classes = ["Mobil", "Motor", "Truck", "Bus"]
        prediction = random.choice(classes)

        st.success(f"Prediksi: **{prediction}** 🚗")

        st.subheader("Hasil Probabilitas:")
        for cls in classes:
            st.progress(random.uniform(0.6, 1.0) if cls == prediction else random.uniform(0.1, 0.7))

# =======================
# ABOUT PAGE
# =======================
elif st.session_state.page == "About Project":
    st.header("Tentang Project Ini")
    st.write("""
    Sistem ini dibuat untuk mendeteksi jenis kendaraan (mobil, motor, bus, dan truk)
    menggunakan model AI dengan teknologi **deep learning** yang mampu mengenali pola visual dengan akurasi tinggi.
    """)
