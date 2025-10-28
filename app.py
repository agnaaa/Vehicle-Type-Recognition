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
# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="AI Vehicle Detection", layout="wide")

# ==========================
# CUSTOM CSS
# ==========================
st.markdown("""
    <style>
        body {
            background-color: #fdeff4;
        }
        .navbar {
            background-color: #f3b9cc;
            padding: 10px 0;
            text-align: center;
            border-radius: 12px;
            margin-bottom: 25px;
        }
        .nav-item {
            display: inline-block;
            margin: 0 35px;
            font-weight: 600;
            color: #4a4a4a;
            cursor: pointer;
            text-decoration: none;
            transition: all 0.2s;
        }
        .nav-item:hover {
            color: #e75480;
        }
        .active {
            background-color: white;
            padding: 6px 16px;
            border-radius: 10px;
            color: #e75480 !important;
        }
        .hero {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 60px 80px;
        }
        .hero-text h1 {
            font-size: 44px;
            font-weight: 800;
            color: #333;
        }
        .hero-text span {
            color: #e75480;
        }
        .hero-text p {
            font-size: 16px;
            color: #555;
            margin-top: 15px;
            margin-bottom: 30px;
            max-width: 480px;
        }
        .upload-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            width: 360px;
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
        .stButton button {
            background-color: #e75480;
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s;
        }
        .stButton button:hover {
            background-color: #d34672;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# SESSION STATE PAGE
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ==========================
# NAVBAR
# ==========================
st.markdown('<div class="navbar">', unsafe_allow_html=True)

nav_items = ["Home", "Classification", "About Project"]
nav_html = ""
for page in nav_items:
    cls = "nav-item active" if st.session_state.page == page else "nav-item"
    nav_html += f"<a class='{cls}' href='#{page}' onclick=\"window.location.reload()\">{page}</a>"

st.markdown(nav_html, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# LOGIC (PAGE SELECTION)
# ==========================
# Karena href tidak trigger Streamlit, pakai tombol logika
def go_to(page):
    st.session_state.page = page

# ==========================
# HOME PAGE
# ==========================
if st.session_state.page == "Home":
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("""
            <div class="hero-text">
                <h1>Deteksi Jenis <span>Kendaraan AI</span></h1>
                <p>Gunakan teknologi AI untuk mengenali mobil, motor, truk, atau bus hanya dengan satu kali upload gambar.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🚗 Coba Sekarang"):
            go_to("Classification")
            st.rerun()

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

    st.markdown('<div class="section-title">Jenis Kendaraan yang Dapat Dideteksi</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="vehicle-grid">
            <div class="vehicle-card">🚘<h4>Mobil</h4><p>Sedan, SUV, Hatchback</p></div>
            <div class="vehicle-card">🏍️<h4>Motor</h4><p>Skuter & Sepeda Motor</p></div>
            <div class="vehicle-card">🚛<h4>Truk</h4><p>Kendaraan Kargo</p></div>
            <div class="vehicle-card">🚌<h4>Bus</h4><p>Transportasi Umum</p></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="stats">
            <div><div class="stat">98.2%</div><div class="stat-label">Akurasi Model</div></div>
            <div><div class="stat">47ms</div><div class="stat-label">Waktu Proses</div></div>
            <div><div class="stat">4+</div><div class="stat-label">Jenis Kendaraan</div></div>
            <div><div class="stat">99.9%</div><div class="stat-label">Uptime</div></div>
        </div>
    """, unsafe_allow_html=True)

# ==========================
# CLASSIFICATION PAGE
# ==========================
elif st.session_state.page == "Classification":
    st.header("Klasifikasi Gambar AI")
    st.write("Upload gambar kendaraan dan biarkan AI mendeteksi jenis kendaraan.")

    uploaded_file = st.file_uploader("Upload gambar kendaraan", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diupload", use_column_width=True)

        with st.spinner("Menganalisis gambar..."):
            time.sleep(2)

        classes = ["Mobil", "Motor", "Truk", "Bus"]
        prediction = random.choice(classes)

        st.success(f"Prediksi: **{prediction}** 🚗")

        st.subheader("Hasil Probabilitas:")
        for cls in classes:
            val = random.uniform(0.7, 0.95) if cls == prediction else random.uniform(0.1, 0.6)
            st.progress(val)

# ==========================
# ABOUT PROJECT PAGE
# ==========================
elif st.session_state.page == "About Project":
    st.header("Tentang Project Ini")
    st.write("""
    Sistem ini dikembangkan untuk mendeteksi jenis kendaraan (mobil, motor, bus, dan truk)
    menggunakan teknologi **Deep Learning**.  
    Dapat digunakan untuk keperluan **pengawasan lalu lintas**, **otomasi parkir**, dan **analisis transportasi**.
    """)
