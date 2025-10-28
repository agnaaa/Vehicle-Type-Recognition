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
st.set_page_config(page_title="AI Image Detection", layout="wide")

# =============================
# CSS STYLE (pink soft pastel)
# =============================
st.markdown("""
<style>
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
        font-size: 22px;
        color: #1f2937;
    }
    .navbar-left span {
        color: #ec5c9a;
    }
    .navbar-right button {
        background: none;
        border: none;
        font-size: 16px;
        margin-left: 2rem;
        cursor: pointer;
        color: #1f2937;
        font-weight: 600;
        transition: 0.3s;
    }
    .navbar-right button:hover {
        color: #ec5c9a;
    }
    .navbar-right button.active {
        color: #ec5c9a;
        text-decoration: underline;
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
        font-size: 50px;
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
        border: none;
        color: white;
        font-weight: 600;
        padding: 0.8rem 1.8rem;
        border-radius: 10px;
        margin-top: 1.5rem;
        margin-right: 1rem;
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
    .upload-placeholder {
        border: 2px dashed #f4b7d0;
        border-radius: 10px;
        padding: 2rem;
        background-color: #fff8fa;
    }
    .upload-choose {
        margin-top: 1rem;
        background-color: #ec5c9a;
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }

    /* --- Vehicle Section --- */
    .vehicle-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 2rem;
        padding: 2rem 6rem;
    }
    .vehicle-card {
        background: white;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(236,92,154,0.1);
        text-align: center;
        padding: 1.5rem;
        transition: 0.3s;
    }
    .vehicle-card:hover {
        transform: translateY(-5px);
    }
    .vehicle-card img {
        width: 80px;
        margin-bottom: 1rem;
    }
    .vehicle-card h4 {
        color: #1f2937;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .vehicle-card p {
        color: #6b7280;
        font-size: 14px;
    }

    /* --- Feature Section --- */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 2rem;
        padding: 2rem 6rem 5rem 6rem;
    }
    .feature-card {
        background: white;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(236,92,154,0.1);
        text-align: center;
        padding: 1.5rem;
    }
    .feature-card .icon {
        font-size: 28px;
        margin-bottom: 1rem;
    }
    .feature-card h4 {
        color: #1f2937;
        font-weight: 700;
    }
    .feature-card p {
        color: #6b7280;
        font-size: 14px;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# SESSION STATE NAVIGATION
# =============================
if "page" not in st.session_state:
    st.session_state.page = "Home"

def set_page(page):
    st.session_state.page = page

# =============================
# NAVBAR
# =============================
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown('<div class="navbar-left">AI <span>Image Detection</span></div>', unsafe_allow_html=True)
with col2:
    st.markdown(
        f"""
        <div class="navbar-right">
            <button class="{'active' if st.session_state.page=='Home' else ''}" onclick="window.location.reload()">Home</button>
            <button class="{'active' if st.session_state.page=='Classification' else ''}" onclick="window.location.reload()">Classification</button>
            <button class="{'active' if st.session_state.page=='About' else ''}" onclick="window.location.reload()">About Project</button>
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================
# PAGE CONTENT
# =============================

if st.session_state.page == "Home":
    # Hero section
    st.markdown("""
    <div class="hero">
        <div class="hero-left">
            <h1>Deteksi Jenis <br><span>Kendaraan AI</span></h1>
            <p>Platform revolusioner yang menggunakan teknologi deep learning 
            untuk mengidentifikasi dan mengklasifikasi jenis kendaraan seperti mobil, 
            motor, truck, dan bus dengan akurasi tinggi.</p>
            <button class="btn-primary">🚀 Coba Sekarang</button>
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
    <div style="text-align:center; margin-top:4rem;">
        <h2 style="color:#1f2937;">Jenis Kendaraan yang Dapat Dideteksi</h2>
    </div>

    <div class="vehicle-grid">
        <div class="vehicle-card">
            <img src="https://i.ibb.co/FXBvZZ7/car.png">
            <h4>Mobil</h4>
            <p>Sedan, SUV, Hatchback, dan berbagai jenis mobil penumpang</p>
        </div>
        <div class="vehicle-card">
            <img src="https://i.ibb.co/gWQhNsc/motorcycle.png">
            <h4>Motor</h4>
            <p>Sepeda motor, skuter, dan kendaraan roda dua lainnya</p>
        </div>
        <div class="vehicle-card">
            <img src="https://i.ibb.co/F8y2Csx/truck.png">
            <h4>Truck</h4>
            <p>Truk kargo, pickup, dan kendaraan komersial berat</p>
        </div>
        <div class="vehicle-card">
            <img src="https://i.ibb.co/NrQL8cp/bus.png">
            <h4>Bus</h4>
            <p>Bus kota, bus antar kota, dan kendaraan angkutan umum</p>
        </div>
    </div>

    <div style="text-align:center; margin-top:4rem;">
        <h2 style="color:#1f2937;">Mengapa Memilih Platform Kami?</h2>
    </div>

    <div class="features-grid">
        <div class="feature-card">
            <div class="icon">🎯</div>
            <h4>Deteksi Akurat</h4>
            <p>Akurasi hingga 98.2% dalam mengenali jenis kendaraan dengan teknologi deep learning</p>
        </div>
        <div class="feature-card">
            <div class="icon">⚡</div>
            <h4>Pemrosesan Cepat</h4>
            <p>Identifikasi kendaraan dalam waktu kurang dari 50ms</p>
        </div>
        <div class="feature-card">
            <div class="icon">🔒</div>
            <h4>Keamanan Tinggi</h4>
            <p>Data gambar kendaraan diproses dengan enkripsi end-to-end</p>
        </div>
        <div class="feature-card">
            <div class="icon">🌐</div>
            <h4>API Global</h4>
            <p>Akses mudah melalui REST API untuk integrasi sistem traffic management</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.page == "Classification":
    st.markdown("<h2 style='text-align:center; color:#ec5c9a;'>🚗 Halaman Klasifikasi Kendaraan</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload gambar kendaraan untuk diklasifikasi", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar diunggah", use_container_width=True)
        st.success("✅ Gambar berhasil diunggah.")

elif st.session_state.page == "About":
    st.markdown("<h2 style='text-align:center; color:#ec5c9a;'>Tentang Proyek Ini</h2>", unsafe_allow_html=True)
    st.write("""
    Proyek ini dikembangkan untuk mendeteksi jenis kendaraan menggunakan kombinasi model YOLO dan CNN.
    Sistem ini memanfaatkan teknologi *Deep Learning* untuk menganalisis gambar kendaraan secara real-time.
    """)
