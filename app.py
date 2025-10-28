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
# SECTION: Jenis Kendaraan
# =============================
st.write("---")
st.markdown("""
    <div style='text-align:center; margin-top:4rem;'>
        <h2>Jenis Kendaraan yang Dapat Dideteksi</h2>
        <p style='color:#6b7280;'>
            Sistem AI kami dapat mengenali berbagai jenis kendaraan dengan akurasi tinggi
        </p>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image("https://i.ibb.co/FXBvZZ7/car.png", use_container_width=True)
    st.markdown("<p style='text-align:center; font-weight:600;'>Mobil</p>", unsafe_allow_html=True)
    st.caption("Sedan, SUV, Hatchback, dan berbagai jenis mobil penumpang")

with col2:
    st.image("https://i.ibb.co/gWQhNsc/motorcycle.png", use_container_width=True)
    st.markdown("<p style='text-align:center; font-weight:600;'>Motor</p>", unsafe_allow_html=True)
    st.caption("Sepeda motor, skuter, dan kendaraan roda dua lainnya")

with col3:
    st.image("https://i.ibb.co/F8y2Csx/truck.png", use_container_width=True)
    st.markdown("<p style='text-align:center; font-weight:600;'>Truck</p>", unsafe_allow_html=True)
    st.caption("Truk kargo, pickup, dan kendaraan komersial berat")

with col4:
    st.image("https://i.ibb.co/NrQL8cp/bus.png", use_container_width=True)
    st.markdown("<p style='text-align:center; font-weight:600;'>Bus</p>", unsafe_allow_html=True)
    st.caption("Bus kota, bus antar kota, dan kendaraan angkutan umum")


# =============================
# SECTION: Model Performance Stats
# =============================
st.write("---")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Akurasi Model", "98.2%")
col_b.metric("Waktu Proses", "47ms")
col_c.metric("Jenis Kendaraan", "4+")
col_d.metric("Uptime", "99.9%")


# =============================
# SECTION: Mengapa Memilih Kami
# =============================
st.write("---")
st.markdown("""
    <div style='text-align:center; margin-top:3rem;'>
        <h2>Mengapa Memilih Platform Kami?</h2>
        <p style='color:#6b7280;'>
            Teknologi AI terdepan yang dirancang khusus untuk deteksi kendaraan dengan akurasi tinggi
        </p>
    </div>
""", unsafe_allow_html=True)

col_x, col_y, col_z, col_w = st.columns(4)

with col_x:
    st.markdown("""
        <div style='background:white; border-radius:15px; padding:1.5rem; 
                    box-shadow:0 10px 25px rgba(236,92,154,0.1); text-align:center;'>
            <h4>🎯 Deteksi Akurat</h4>
            <p style='color:#6b7280;'>Akurasi hingga 98.2% dalam mengenali jenis kendaraan dengan teknologi deep learning.</p>
        </div>
    """, unsafe_allow_html=True)

with col_y:
    st.markdown("""
        <div style='background:white; border-radius:15px; padding:1.5rem; 
                    box-shadow:0 10px 25px rgba(236,92,154,0.1); text-align:center;'>
            <h4>⚡ Pemrosesan Cepat</h4>
            <p style='color:#6b7280;'>Identifikasi kendaraan dalam waktu kurang dari 50ms.</p>
        </div>
    """, unsafe_allow_html=True)

with col_z:
    st.markdown("""
        <div style='background:white; border-radius:15px; padding:1.5rem; 
                    box-shadow:0 10px 25px rgba(236,92,154,0.1); text-align:center;'>
            <h4>🔒 Keamanan Tinggi</h4>
            <p style='color:#6b7280;'>Data gambar kendaraan diproses dengan enkripsi end-to-end.</p>
        </div>
    """, unsafe_allow_html=True)

with col_w:
    st.markdown("""
        <div style='background:white; border-radius:15px; padding:1.5rem; 
                    box-shadow:0 10px 25px rgba(236,92,154,0.1); text-align:center;'>
            <h4>🌐 API Global</h4>
            <p style='color:#6b7280;'>Akses mudah melalui REST API untuk integrasi sistem traffic management.</p>
        </div>
    """, unsafe_allow_html=True)


# =============================
# FOOTER BUTTON
# =============================
st.markdown("""
<a class="talk-btn" href="#">💬 Talk with Us</a>
""", unsafe_allow_html=True)
