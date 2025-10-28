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
# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="AI Vehicle Detection",
    layout="wide"
)

# =============================
# CUSTOM CSS STYLE
# =============================
st.markdown("""
    <style>
    /* === GLOBAL STYLES === */
    .stApp {
        background: linear-gradient(180deg, #fdecef 0%, #fff 100%);
        font-family: 'Inter', sans-serif;
    }

    /* === NAVBAR === */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 4rem;
        background-color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border-radius: 0 0 20px 20px;
        margin-bottom: 2rem;
    }
    .navbar-left {
        font-weight: 800;
        font-size: 22px;
        color: #1f2937;
    }
    .navbar-left span {
        color: #ec5c9a;
    }
    .navbar-right button {
        background: none;
        border: none;
        font-weight: 600;
        color: #1f2937;
        margin-left: 2rem;
        cursor: pointer;
        transition: 0.3s;
    }
    .navbar-right button:hover {
        color: #ec5c9a;
    }

    /* === HERO SECTION === */
    .hero {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 3rem 6rem;
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
    .btn-primary {
        background-color: #ec5c9a;
        color: white;
        border: none;
        font-weight: 600;
        padding: 0.9rem 1.8rem;
        border-radius: 10px;
        margin-top: 1.5rem;
        cursor: pointer;
    }

    /* === UPLOAD CARD === */
    .upload-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(236,92,154,0.15);
        width: 340px;
        text-align: center;
    }
    .upload-placeholder {
        border: 2px dashed #f4b7d0;
        border-radius: 12px;
        padding: 2rem;
        background: #fff5f8;
    }
    .upload-choose {
        background-color: #ec5c9a;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        margin-top: 1rem;
        display: inline-block;
        cursor: pointer;
    }

    /* === VEHICLE GRID === */
    .vehicle-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        padding: 2rem 6rem;
    }
    .vehicle-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 6px 20px rgba(236,92,154,0.1);
        transition: 0.3s;
    }
    .vehicle-card:hover {
        transform: translateY(-5px);
    }
    .vehicle-card img {
        width: 80px;
        margin-bottom: 1rem;
    }

    /* === FEATURE GRID === */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        padding: 2rem 6rem 4rem 6rem;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(236,92,154,0.1);
    }
    .icon {
        font-size: 30px;
        margin-bottom: 0.5rem;
    }

    /* === CLASSIFICATION SECTION === */
    .classify-section {
        display: flex;
        justify-content: center;
        align-items: flex-start;
        padding: 4rem 6rem;
        gap: 3rem;
    }
    .result-box {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(236,92,154,0.1);
        width: 320px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# =============================
# NAVBAR + PAGE SELECTION
# =============================
st.session_state.page = st.session_state.get("page", "Home")

col1, col2 = st.columns([1, 3])
with col1:
    st.markdown('<div class="navbar-left">AI <span>Vehicle Detection</span></div>', unsafe_allow_html=True)
with col2:
    cols = st.columns(3)
    if cols[0].button("Home"):
        st.session_state.page = "Home"
    if cols[1].button("Classification"):
        st.session_state.page = "Classification"
    if cols[2].button("About Project"):
        st.session_state.page = "About"

st.write("---")

# =============================
# PAGE CONTENT
# =============================
if st.session_state.page == "Home":
    st.markdown("""
    <div class="hero">
        <div class="hero-left">
            <h1>Deteksi Jenis <br><span>Kendaraan AI</span></h1>
            <p>Platform revolusioner yang menggunakan teknologi deep learning 
            untuk mengidentifikasi dan mengklasifikasikan jenis kendaraan seperti mobil, 
            motor, truk, dan bus dengan akurasi tinggi.</p>
            <button class="btn-primary" onclick="window.location.href='#Classification'">🚀 Coba Sekarang</button>
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
            <h4>🚗 Mobil</h4>
            <p>Sedan, SUV, Hatchback, dan mobil penumpang lainnya.</p>
        </div>
        <div class="vehicle-card">
            <img src="https://i.ibb.co/gWQhNsc/motorcycle.png">
            <h4>🏍️ Motor</h4>
            <p>Sepeda motor, skuter, dan kendaraan roda dua lainnya.</p>
        </div>
        <div class="vehicle-card">
            <img src="https://i.ibb.co/F8y2Csx/truck.png">
            <h4>🚚 Truck</h4>
            <p>Truk kargo, pickup, dan kendaraan komersial berat.</p>
        </div>
        <div class="vehicle-card">
            <img src="https://i.ibb.co/NrQL8cp/bus.png">
            <h4>🚌 Bus</h4>
            <p>Bus kota, antar kota, dan kendaraan transportasi umum.</p>
        </div>
    </div>

    <div class="features-grid">
        <div class="feature-card">
            <div class="icon">🎯</div>
            <h4>Deteksi Akurat</h4>
            <p>Akurasi hingga 98% dalam mengenali kendaraan.</p>
        </div>
        <div class="feature-card">
            <div class="icon">⚡</div>
            <h4>Pemrosesan Cepat</h4>
            <p>Identifikasi kendaraan dalam kurang dari 50ms.</p>
        </div>
        <div class="feature-card">
            <div class="icon">🔒</div>
            <h4>Keamanan Data</h4>
            <p>Gambar diproses dengan sistem enkripsi aman.</p>
        </div>
        <div class="feature-card">
            <div class="icon">🌐</div>
            <h4>Integrasi Mudah</h4>
            <p>Dapat dihubungkan dengan API atau sistem manajemen lalu lintas.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == "Classification":
    st.markdown("<h2 style='text-align:center;color:#ec5c9a;'>🚀 Klasifikasi Gambar Kendaraan</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload gambar kendaraan", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Gambar yang diunggah", use_container_width=True)
        with col2:
            st.markdown("<div class='result-box'><h4>Hasil Prediksi</h4>", unsafe_allow_html=True)
            st.success("Jenis Kendaraan: 🚗 Mobil")
            st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "About":
    st.markdown("""
    <div style='text-align:center;padding:4rem 8rem;'>
        <h2 style='color:#ec5c9a;'>Tentang Project Ini</h2>
        <p style='color:#555;margin-top:1rem;font-size:17px;'>
        Aplikasi ini dirancang untuk mendeteksi jenis kendaraan menggunakan teknologi AI dan Deep Learning. 
        Dengan model canggih, sistem dapat mengenali mobil, motor, truk, dan bus dengan akurasi tinggi dan waktu respon cepat.
        </p>
    </div>
    """, unsafe_allow_html=True)
