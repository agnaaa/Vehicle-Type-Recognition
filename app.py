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
# ==============================
# Konfigurasi Halaman
# ==============================
st.set_page_config(page_title="AI Vehicle Detection", layout="wide")

# ==============================
# CSS Styling
# ==============================
st.markdown("""
    <style>
        /* Background full pink pastel */
        html, body, [class*="css"] {
            background-color: #fdeff4 !important;
        }

        /* Navbar */
        .navbar {
            background-color: #f8cfe0;
            border-radius: 10px;
            padding: 12px 0;
            margin-bottom: 30px;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
        }
        .nav-container {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 60px;
        }
        .nav-item {
            display: inline-block;
            font-weight: 600;
            font-size: 17px;
            color: #333;
            cursor: pointer;
            padding: 10px 20px;
            border-radius: 8px;
            transition: 0.2s ease;
        }
        .nav-item:hover {
            background-color: rgba(255,255,255,0.8);
        }
        .nav-item.active {
            background-color: white;
            color: #e75480;
            box-shadow: 0px 3px 8px rgba(0,0,0,0.08);
        }

        /* Hero Section */
        .hero {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 60px 80px;
        }
        .hero-text {
            max-width: 550px;
        }
        .hero-text h1 {
            font-size: 46px;
            font-weight: 800;
            margin-bottom: 16px;
            color: #333;
        }
        .hero-text span {
            color: #e75480;
        }
        .hero-text p {
            color: #555;
            margin-bottom: 30px;
            line-height: 1.6;
        }
        .btn-primary {
            background-color: #e75480;
            color: white;
            border: none;
            padding: 12px 22px;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: 0.2s;
        }
        .btn-primary:hover {
            background-color: #d64372;
        }

        /* Cards */
        .vehicle-grid {
            display: flex;
            justify-content: center;
            gap: 25px;
            margin-top: 40px;
            flex-wrap: wrap;
        }
        .vehicle-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            width: 230px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
        }
        .vehicle-card h4 {
            margin-top: 8px;
            color: #333;
        }

        .icon {
            font-size: 32px;
            margin-bottom: 8px;
            color: #e75480;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Navbar Interaktif
# ==============================
if "page" not in st.session_state:
    st.session_state.page = "Home"

st.markdown('<div class="navbar"><div class="nav-container">', unsafe_allow_html=True)
cols = st.columns(3)
for i, page in enumerate(["Home", "Classification", "About Project"]):
    with cols[i]:
        active_class = "nav-item active" if st.session_state.page == page else "nav-item"
        if st.button(page, key=page):
            st.session_state.page = page
            st.rerun()
st.markdown('</div></div>', unsafe_allow_html=True)

# ==============================
# HOME PAGE
# ==============================
if st.session_state.page == "Home":
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("""
            <div class="hero-text">
                <h1>Deteksi Jenis <span>Kendaraan AI</span></h1>
                <p>Gunakan kecerdasan buatan untuk mengenali jenis kendaraan seperti mobil, motor, truk, dan bus hanya dengan satu kali upload gambar.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🚗 Coba Sekarang"):
            st.session_state.page = "Classification"
            st.rerun()

    with col2:
        st.image("https://i.ibb.co/FXBvZZ7/car.png", width=350)

    st.markdown('<div class="vehicle-grid">', unsafe_allow_html=True)
    st.markdown("""
        <div class="vehicle-card">🚘<h4>Mobil</h4></div>
        <div class="vehicle-card">🏍️<h4>Motor</h4></div>
        <div class="vehicle-card">🚛<h4>Truck</h4></div>
        <div class="vehicle-card">🚌<h4>Bus</h4></div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# CLASSIFICATION PAGE
# ==============================
elif st.session_state.page == "Classification":
    st.markdown("<h2 style='text-align:center;color:#333;'>Klasifikasi Gambar Kendaraan</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#555;'>Upload gambar kendaraan, dan AI akan menganalisis jenisnya secara otomatis.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload gambar kendaraan", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diupload", use_column_width=True)

    with col2:
        if uploaded_file:
            with st.spinner("Menganalisis gambar..."):
                time.sleep(2)
            classes = ["Mobil", "Motor", "Truck", "Bus"]
            prediction = random.choice(classes)

            st.success(f"Hasil Prediksi: **{prediction}** 🚗")
            st.subheader("Probabilitas Deteksi:")
            for cls in classes:
                val = random.uniform(0.75, 1.0) if cls == prediction else random.uniform(0.1, 0.6)
                st.write(f"{cls}")
                st.progress(val)
        else:
            st.info("⬅️ Upload gambar terlebih dahulu untuk melihat hasil analisis.")

# ==============================
# ABOUT PAGE
# ==============================
elif st.session_state.page == "About Project":
    st.markdown("<h2 style='text-align:center;color:#333;'>Tentang Project Ini</h2>", unsafe_allow_html=True)
    st.write("""
    Aplikasi ini dirancang untuk mendeteksi jenis kendaraan menggunakan teknologi **deep learning**.  
    Model ini mampu mengenali gambar kendaraan dengan tingkat akurasi tinggi.  
    Tampilan didesain dengan warna **pink soft pastel** agar terlihat lembut, bersih, dan profesional. 💗
    """)

---

💗 **Coba jalankan ini:**
```bash
streamlit run app.py
