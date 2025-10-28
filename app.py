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
# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="AI Image Detection",
    layout="wide"
)

# =============================
# CUSTOM CSS STYLE
# =============================
st.markdown("""
    <style>
    /* --- Main background --- */
    .stApp {
        background: linear-gradient(180deg, #fdeef2 0%, #ffffff 100%);
        font-family: 'Inter', sans-serif;
    }

    /* --- Navbar --- */
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
    .navbar-right a {
        margin-left: 2rem;
        text-decoration: none;
        font-weight: 600;
        color: #1f2937;
        transition: 0.3s;
    }
    .navbar-right a:hover {
        color: #ec5c9a;
    }

    /* --- Hero Section --- */
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

    /* --- Buttons --- */
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

    /* --- Upload Card --- */
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

    /* --- Footer Button --- */
    .talk-btn {
        position: fixed;
        bottom: 25px;
        right: 25px;
        background-color: #ec5c9a;
        color: white;
        padding: 12px 24px;
        border-radius: 30px;
        font-weight: 600;
        text-decoration: none;
        box-shadow: 0 4px 15px rgba(236,92,154,0.4);
    }
    .talk-btn:hover {
        background-color: #e34c8f;
    }
    </style>
""", unsafe_allow_html=True)

# =============================
# NAVBAR
# =============================
st.markdown("""
<div class="navbar">
    <div class="navbar-left">AI <span>Image Detection</span></div>
    <div class="navbar-right">
        <a href="#">Home</a>
        <a href="#">Classification</a>
        <a href="#">Model Performance</a>
        <a href="#">Model Info</a>
        <a href="#">About Project</a>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================
# HERO SECTION
# =============================
st.markdown("""
<div class="hero">
    <div class="hero-left">
        <h1>Deteksi Jenis <br><span>Kendaraan AI</span></h1>
        <p>Platform revolusioner yang menggunakan teknologi deep learning 
        untuk mengidentifikasi dan mengklasifikasi jenis kendaraan seperti mobil, 
        motor, truck, dan bus dengan akurasi tinggi.</p>
        <button class="btn-primary">🚀 Coba Sekarang</button>
        <button class="btn-outline">📘 Pelajari Lebih Lanjut</button>
    </div>

    <div class="upload-card">
        <h4>Demo Cepat</h4>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload gambar kendaraan untuk analisis", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar kendaraan diunggah", use_container_width=True)
    st.success("✅ Gambar berhasil diunggah (Demo).")

st.markdown("""
    </div>
</div>
""", unsafe_allow_html=True)

# =============================
# FOOTER BUTTON
# =============================
st.markdown("""
<a class="talk-btn" href="#">💬 Talk with Us</a>
""", unsafe_allow_html=True)

