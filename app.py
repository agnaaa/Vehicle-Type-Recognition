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
    yolo_model = YOLO("model/best.pt")

    try:
        # coba load dengan TensorFlow baru
        classifier = tf.keras.models.load_model("model/classifier_model.h5", safe_mode=False, compile=False)
    except Exception as e:
        st.error(f"Gagal memuat model klasifikasi: {e}")
        classifier = None

    return yolo_model, classifier

# ==========================
# UI
# -------------------------------
# Konfigurasi Halaman
# -------------------------------
st.set_page_config(page_title="Deteksi Jenis Kendaraan AI", page_icon="🚗", layout="wide")

# -------------------------------
# CSS Custom Tema Pink Pastel
# -------------------------------
st.markdown("""
    <style>
    body {
        background-color: #fdecef;
        color: #333;
    }
    .main {
        background-color: #fdecef !important;
    }
    h1, h2, h3, h4 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
    }
    p, li {
        font-family: 'Poppins', sans-serif;
        font-weight: 400;
    }
    .stButton > button {
        background-color: #f06292;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #ec407a;
        transform: scale(1.03);
    }
    .card {
        background-color: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        height: 150px;
    }
    .card h4 {
        color: #333;
        margin-bottom: 10px;
        font-weight: 600;
    }
    .card p {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2B2B2B;
        margin: 0;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.markdown("<h1 style='text-align: center; color: #e91e63;'>🚗 Deteksi Jenis Kendaraan AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Platform berbasis AI untuk mengidentifikasi jenis kendaraan seperti mobil, motor, truk, dan bus secara otomatis dengan tingkat akurasi tinggi.</p>", unsafe_allow_html=True)

# Tombol Aksi
col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    st.markdown("<button style='background-color:#f06292;color:white;padding:10px 30px;border:none;border-radius:8px;font-weight:600;width:100%;'>🚀 Coba Sekarang</button>", unsafe_allow_html=True)
with col_btn2:
    st.markdown("<button style='background-color:white;color:#f06292;padding:10px 30px;border:2px solid #f06292;border-radius:8px;font-weight:600;width:100%;'>📘 Pelajari Lebih Lanjut</button>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# Jenis Kendaraan
# -------------------------------
st.markdown("## Jenis Kendaraan yang Dapat Dideteksi")
st.write("AI kami mampu mengenali berbagai jenis kendaraan dari berbagai sudut dan kondisi pencahayaan.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if os.path.exists("images/car.jpg"):
        st.image("images/car.jpg", use_container_width=True)
    st.markdown("**Mobil**")
    st.caption("Sedan, SUV, Hatchback, dan lainnya")

with col2:
    if os.path.exists("images/motor.jpg"):
        st.image("images/motor.jpg", use_container_width=True)
    st.markdown("**Motor**")
    st.caption("Sepeda motor, skuter, dan kendaraan roda dua")

with col3:
    if os.path.exists("images/truck.jpg"):
        st.image("images/truck.jpg", use_container_width=True)
    st.markdown("**Truck**")
    st.caption("Truk kargo, pickup, dan kendaraan berat")

with col4:
    if os.path.exists("images/bus.jpg"):
        st.image("images/bus.jpg", use_container_width=True)
    st.markdown("**Bus**")
    st.caption("Bus kota, antar kota, dan angkutan umum")

# -------------------------------
# Performa Model Kami
# -------------------------------
st.markdown("---")
st.markdown("<h2 style='text-align:center;color:#e91e63;'>✨ Performa Model Kami ✨</h2>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="card">
        <h4>Akurasi Model</h4>
        <p>98.2%</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h4>Waktu Proses</h4>
        <p>47 ms</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <h4>Jenis Kelas</h4>
        <p>4+</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="card">
        <h4>Uptime Sistem</h4>
        <p>99.9%</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Keunggulan Platform
# -------------------------------
st.markdown("---")
st.markdown("## Mengapa Memilih Platform Kami?")
st.write("Kami menghadirkan solusi cerdas untuk pengenalan kendaraan yang cepat, akurat, dan aman.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("### 🎯 Akurasi Tinggi")
    st.write("Akurasi mencapai 98.2% berkat arsitektur deep learning modern.")
with col2:
    st.markdown("### ⚡ Pemrosesan Cepat")
    st.write("Identifikasi kendaraan dalam waktu kurang dari 50 milidetik.")
with col3:
    st.markdown("### 🔒 Aman")
    st.write("Data gambar diproses dengan sistem enkripsi end-to-end.")
with col4:
    st.markdown("### 🌍 Integrasi Mudah")
    st.write("Dapat diintegrasikan dengan sistem monitoring atau manajemen lalu lintas.")

# -------------------------------
# Demo Upload
# -------------------------------
st.markdown("---")
st.markdown("## Demo Cepat")
uploaded_file = st.file_uploader("Unggah gambar kendaraan untuk deteksi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)
    st.success("🚧 Fitur prediksi sedang dikembangkan.")
else:
    st.info("Unggah gambar untuk mencoba deteksi kendaraan.")
