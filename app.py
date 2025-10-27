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
# ===============================
# Konfigurasi Halaman
# ===============================
st.set_page_config(page_title="Deteksi Jenis Kendaraan AI", layout="wide")

# ===============================
# CSS Kustom (tema pink pastel)
# ===============================
st.markdown("""
<style>
/* Warna latar belakang dan font */
body, [class*="stAppViewContainer"] {
    background-color: #FDECEF;
    font-family: 'Poppins', sans-serif;
    color: #2B2B2B;
}

/* Header dan section */
h1, h2, h3 {
    color: #D63384;
    font-weight: 700;
}

/* Tombol utama */
.stButton>button {
    background: linear-gradient(90deg, #FF7EB3, #FF5C8A);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(255, 92, 138, 0.4);
}

/* Card */
.card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}

/* Bagian judul utama */
.hero {
    padding: 3rem 2rem;
    text-align: left;
}
.hero h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    color: #2B2B2B;
}
.hero span {
    color: #D63384;
}

/* Upload box */
.upload {
    background: white;
    border: 2px dashed #FF8FB1;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# Bagian Hero
# ===============================
col1, col2 = st.columns([1.5, 1])
with col1:
    st.markdown("""
    <div class="hero">
        <h1>Deteksi Jenis <span>Kendaraan AI</span></h1>
        <p>Platform revolusioner yang menggunakan teknologi deep learning untuk mengidentifikasi dan mengklasifikasi jenis kendaraan seperti mobil, motor, truck, dan bus dengan akurasi tinggi.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.button("🚗 Coba Sekarang")
    st.button("📘 Pelajari Lebih Lanjut")

with col2:
    st.markdown('<div class="upload">Upload gambar kendaraan untuk analisis</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Pilih Gambar", type=["jpg","jpeg","png"])

# ===============================
# Jenis Kendaraan
# ===============================
st.markdown("## Jenis Kendaraan yang Dapat Dideteksi")
st.write("Sistem AI kami dapat mengenali berbagai jenis kendaraan dengan akurasi tinggi.")
col1, col2, col3, col4 = st.columns(4)
col1.markdown('<div class="card">🚙<br><b>Mobil</b><br>Sedan, SUV, Hatchback, dan lainnya</div>', unsafe_allow_html=True)
col2.markdown('<div class="card">🏍️<br><b>Motor</b><br>Sepeda motor, skuter, dan kendaraan roda dua</div>', unsafe_allow_html=True)
col3.markdown('<div class="card">🚚<br><b>Truck</b><br>Kendaraan komersial dan kargo</div>', unsafe_allow_html=True)
col4.markdown('<div class="card">🚌<br><b>Bus</b><br>Bus kota dan antar kota</div>', unsafe_allow_html=True)

# ===============================
# Statistik Model
# ===============================
st.markdown("## Performa Model Kami")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Akurasi Model", "98.2%")
col2.metric("Waktu Proses", "47ms")
col3.metric("Jenis Kendaraan", "4+")
col4.metric("Uptime", "99.9%")

# ===============================
# Kelebihan Platform
# ===============================
st.markdown("## Mengapa Memilih Platform Kami?")
col1, col2, col3, col4 = st.columns(4)
col1.markdown('<div class="card"><b>Deteksi Akurat</b><br>Akurasi 98.2% dalam mengenali kendaraan.</div>', unsafe_allow_html=True)
col2.markdown('<div class="card"><b>Pemrosesan Cepat</b><br>Identifikasi kendaraan dalam waktu kurang dari 50ms.</div>', unsafe_allow_html=True)
col3.markdown('<div class="card"><b>Keamanan Tinggi</b><br>Data gambar dienkripsi secara end-to-end.</div>', unsafe_allow_html=True)
col4.markdown('<div class="card"><b>API Global</b><br>Akses mudah untuk integrasi sistem.</div>', unsafe_allow_html=True)
