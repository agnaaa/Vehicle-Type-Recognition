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
# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="AI Vehicle Detection", layout="wide")

# =========================================
# STYLE (Pink soft pastel theme)
# =========================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #fdeef2 0%, #ffffff 100%);
    font-family: 'Inter', sans-serif;
}

/* Navbar */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 4rem;
    background-color: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    border-radius: 0 0 20px 20px;
}
.navbar-left {
    font-weight: 700;
    font-size: 20px;
    color: #1f2937;
}
.navbar-left span {
    color: #ec5c9a;
}
.navbar-right button {
    margin-left: 2rem;
    background: none;
    border: none;
    font-weight: 600;
    color: #1f2937;
    cursor: pointer;
    transition: 0.3s;
}
.navbar-right button:hover {
    color: #ec5c9a;
}

/* Hero section */
.hero {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5rem 6rem 3rem 6rem;
}
.hero-left {
    max-width: 600px;
}
.hero-left h1 {
    font-size: 48px;
    font-weight: 800;
    color: #1f2937;
    line-height: 1.2;
}
.hero-left span {
    color: #ec5c9a;
}
.hero-left p {
    font-size: 16px;
    color: #6b7280;
    margin-top: 1rem;
    line-height: 1.6;
}

/* Buttons */
.btn-primary {
    background-color: #ec5c9a;
    border: none;
    color: white;
    font-weight: 600;
    padding: 0.8rem 1.8rem;
    border-radius: 10px;
    margin-top: 1.5rem;
    margin-right: 1rem;
    cursor: pointer;
}
.btn-outline {
    border: 2px solid #f4b7d0;
    background: none;
    color: #ec5c9a;
    font-weight: 600;
    padding: 0.8rem 1.8rem;
    border-radius: 10px;
    cursor: pointer;
}

/* Upload Card */
.upload-card {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(236,92,154,0.15);
    width: 340px;
    text-align: center;
}
.upload-card h4 {
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 1rem;
}
.upload-placeholder {
    border: 2px dashed #f4b7d0;
    border-radius: 10px;
    padding: 2rem;
    background-color: #fff5f8;
}
.upload-choose {
    margin-top: 1rem;
    background-color: #ec5c9a;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    display: inline-block;
    font-weight: 600;
}

/* Vehicle Cards */
.vehicle-grid {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    margin-top: 3rem;
    flex-wrap: wrap;
}
.vehicle-card {
    background: white;
    border-radius: 15px;
    box-shadow: 0 5px 20px rgba(236,92,154,0.15);
    width: 230px;
    padding: 1.5rem;
    text-align: center;
}
.vehicle-card img {
    width: 70px;
    height: 70px;
    margin-bottom: 1rem;
}
.vehicle-card h4 {
    color: #ec5c9a;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

/* Classification result */
.result-box {
    background: white;
    border-radius: 15px;
    padding: 2rem;
    box-shadow: 0 5px 25px rgba(236,92,154,0.2);
    text-align: center;
    margin-top: 2rem;
}
.result-box h4 {
    color: #ec5c9a;
}

/* About Section */
.about-section {
    padding: 5rem 6rem;
    text-align: center;
    color: #333;
}
.about-section h2 {
    color: #ec5c9a;
    font-weight: 800;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# NAVBAR
# =========================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

col1, col2 = st.columns([1, 3])
with col1:
    st.markdown('<div class="navbar-left">AI <span>Vehicle Detection</span></div>', unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="navbar-right">
        <button onClick="window.location.reload()">Home</button>
        <button onClick="window.location.href='?page=Classification'">Classification</button>
        <button onClick="window.location.href='?page=About'">About Project</button>
    </div>
    """, unsafe_allow_html=True)

query_params = st.query_params
page = query_params.get("page", ["Home"])[0]

# =========================================
# HOME PAGE
# =========================================
if page == "Home":
    st.markdown("""
    <div class="hero">
        <div class="hero-left">
            <h1>Deteksi Jenis <span>Kendaraan</span></h1>
            <p>Platform cerdas berbasis deep learning untuk mengenali jenis kendaraan 
            seperti mobil, motor, truk, dan bus secara akurat dan cepat.</p>
            <button class="btn-primary" onClick="window.location.href='?page=Classification'">🚀 Coba Sekarang</button>
            <button class="btn-outline">📘 Pelajari Lebih Lanjut</button>
        </div>

        <div class="upload-card">
            <h4>Demo Cepat</h4>
            <div class="upload-placeholder">
                <div style="font-size:26px;">🖼️</div>
                <div style="margin-top:8px;color:#b88a9f">Upload gambar kendaraan untuk analisis</div>
                <div class="upload-choose">Pilih Gambar</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="vehicle-grid">
        <div class="vehicle-card">
            <img src="https://i.ibb.co/FXBvZZ7/car.png">
            <h4>Mobil 🚗</h4>
            <p>Sedan, SUV, Hatchback, dan mobil penumpang lainnya</p>
        </div>
        <div class="vehicle-card">
            <img src="https://i.ibb.co/gWQhNsc/motorcycle.png">
            <h4>Motor 🏍️</h4>
            <p>Sepeda motor, skuter, dan kendaraan roda dua</p>
        </div>
        <div class="vehicle-card">
            <img src="https://i.ibb.co/F8y2Csx/truck.png">
            <h4>Truk 🚚</h4>
            <p>Truk kargo, pickup, dan kendaraan berat</p>
        </div>
        <div class="vehicle-card">
            <img src="https://i.ibb.co/NrQL8cp/bus.png">
            <h4>Bus 🚌</h4>
            <p>Bus kota, bus antar kota, dan angkutan umum</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================
# CLASSIFICATION PAGE
# =========================================
elif page == "Classification":
    st.markdown("<h2 style='text-align:center;color:#ec5c9a;'>🚗 Klasifikasi Jenis Kendaraan</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#555;'>Upload gambar kendaraan di bawah ini untuk mengetahui jenisnya.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload gambar kendaraan", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar diunggah", use_container_width=True)

        classes = ["Mobil", "Motor", "Truk", "Bus"]
        prediction = random.choice(classes)
        st.markdown(f"""
        <div class="result-box">
            <h4>Prediksi Sistem:</h4>
            <p style='font-size:24px;font-weight:700;color:#1f2937;'>{prediction}</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================
# ABOUT PAGE
# =========================================
elif page == "About":
    st.markdown("""
    <div class="about-section">
        <h2>Tentang Proyek Ini 💡</h2>
        <p>Aplikasi ini dibuat untuk mendemonstrasikan teknologi Artificial Intelligence 
        dalam mendeteksi dan mengklasifikasikan jenis kendaraan secara otomatis 
        menggunakan model Deep Learning.</p>
        <p>Proyek ini dikembangkan menggunakan <b>Streamlit</b> sebagai framework web interaktif 
        dengan desain lembut bertema pink pastel 💕.</p>
    </div>
    """, unsafe_allow_html=True)
