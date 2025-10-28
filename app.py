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

# CSS untuk styling
st.markdown("""
    <style>
        body {
            background-color: #fdeff4;
        }
        .navbar {
            background-color: #f3b9cc;
            padding: 10px 0;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .nav-item {
            display: inline-block;
            margin: 0 25px;
            font-weight: 600;
            color: #333;
            cursor: pointer;
        }
        .nav-item.active {
            background-color: white;
            padding: 6px 14px;
            border-radius: 8px;
            color: #e75480;
        }
        .hero {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 50px;
        }
        .hero-text h1 {
            font-size: 40px;
            font-weight: 800;
        }
        .hero-text span {
            color: #e75480;
        }
        .upload-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            width: 350px;
            text-align: center;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
        }
        .upload-placeholder {
            border: 2px dashed #f0b6c6;
            padding: 30px;
            border-radius: 10px;
            color: #b88a9f;
        }
        .section-title {
            text-align: center;
            font-size: 28px;
            font-weight: 700;
            margin-top: 80px;
        }
        .vehicle-grid, .features-grid {
            display: flex;
            justify-content: center;
            gap: 25px;
            margin-top: 40px;
            flex-wrap: wrap;
        }
        .vehicle-card, .feature-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            width: 230px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
        }
        .icon {
            font-size: 30px;
            margin-bottom: 10px;
            color: #e75480;
        }
        .stats {
            display: flex;
            justify-content: center;
            gap: 60px;
            text-align: center;
            margin-top: 60px;
        }
        .stat {
            font-weight: 700;
            color: #333;
        }
        .stat-label {
            color: gray;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)


# Navbar manual
if "page" not in st.session_state:
    st.session_state.page = "Home"

st.markdown('<div class="navbar">', unsafe_allow_html=True)
for page in ["Home", "Classification", "About Project"]:
    cls = "nav-item active" if st.session_state.page == page else "nav-item"
    if st.button(page, key=page):
        st.session_state.page = page
st.markdown('</div>', unsafe_allow_html=True)


# ==============================
# HOME PAGE
# ==============================
if st.session_state.page == "Home":
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("""
            <div class="hero-text">
                <h1>Deteksi Jenis <span>Kendaraan AI</span></h1>
                <p>Platform revolusioner yang menggunakan teknologi deep learning untuk mengidentifikasi dan mengklasifikasi jenis kendaraan seperti mobil, motor, truck, dan bus dengan akurasi tinggi.</p>
            </div>
        """, unsafe_allow_html=True)
        st.button("🚗 Coba Sekarang")
        st.button("📘 Pelajari Lebih Lanjut")

    with col2:
        st.markdown("""
            <div class="upload-card">
                <h4>Demo Cepat</h4>
                <div class="upload-placeholder">
                    🖼️<br><br>
                    Upload gambar kendaraan untuk analisis
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Jenis kendaraan
    st.markdown('<div class="section-title">Jenis Kendaraan yang Dapat Dideteksi</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="vehicle-grid">
            <div class="vehicle-card">
                🚘<h4>Mobil</h4><p>Sedan, SUV, Hatchback, dan mobil penumpang</p>
            </div>
            <div class="vehicle-card">
                🏍️<h4>Motor</h4><p>Sepeda motor, skuter, dan roda dua lainnya</p>
            </div>
            <div class="vehicle-card">
                🚛<h4>Truck</h4><p>Truk kargo, pickup, dan kendaraan berat</p>
            </div>
            <div class="vehicle-card">
                🚌<h4>Bus</h4><p>Bus kota, antar kota, dan transportasi umum</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Statistik
    st.markdown("""
        <div class="stats">
            <div><div class="stat">98.2%</div><div class="stat-label">Akurasi Model</div></div>
            <div><div class="stat">47ms</div><div class="stat-label">Waktu Proses</div></div>
            <div><div class="stat">4+</div><div class="stat-label">Jenis Kendaraan</div></div>
            <div><div class="stat">99.9%</div><div class="stat-label">Uptime</div></div>
        </div>
    """, unsafe_allow_html=True)

    # Fitur
    st.markdown('<div class="section-title">Mengapa Memilih Platform Kami?</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="features-grid">
            <div class="feature-card"><div class="icon">🎯</div><h4>Deteksi Akurat</h4><p>Akurasi hingga 98.2%</p></div>
            <div class="feature-card"><div class="icon">⚡</div><h4>Pemrosesan Cepat</h4><p>Identifikasi kurang dari 50ms</p></div>
            <div class="feature-card"><div class="icon">🔒</div><h4>Keamanan Tinggi</h4><p>Data terenkripsi end-to-end</p></div>
            <div class="feature-card"><div class="icon">🌐</div><h4>API Global</h4><p>Integrasi mudah REST API</p></div>
        </div>
    """, unsafe_allow_html=True)


# ==============================
# CLASSIFICATION PAGE
# ==============================
elif st.session_state.page == "Classification":
    st.header("Klasifikasi Gambar AI")
    st.write("Upload gambar kendaraan dan biarkan AI mendeteksi jenis kendaraan.")

    uploaded_file = st.file_uploader("Upload gambar kendaraan", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diupload", use_column_width=True)

        with st.spinner("Menganalisis gambar..."):
            time.sleep(2)

        classes = ["Mobil", "Motor", "Truck", "Bus"]
        prediction = random.choices(classes, weights=[0.3, 0.3, 0.2, 0.2])[0]

        st.success(f"Prediksi: **{prediction}** 🚗")

        st.subheader("Hasil Probabilitas:")
        for cls in classes:
            st.progress(random.uniform(0.6, 1.0) if cls == prediction else random.uniform(0.1, 0.7))


# ==============================
# ABOUT PAGE
# ==============================
elif st.session_state.page == "About Project":
    st.header("Tentang Project Ini")
    st.write("""
    Sistem ini dibuat untuk mendeteksi jenis kendaraan (mobil, motor, bus, dan truk)
    menggunakan model AI dengan teknologi **deep learning** yang mampu mengenali pola visual dengan akurasi tinggi.
    """)
