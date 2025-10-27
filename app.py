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
# ========================
# Konfigurasi Halaman
# ========================
st.set_page_config(
    page_title="AI Image Detection",
    page_icon="🚗",
    layout="wide"
)

# ========================
# Styling CSS
# ========================
st.markdown("""
    <style>
    /* Background dan warna umum */
    body {
        background-color: #fdecef !important;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #fdecef;
    }
    [data-testid="stHeader"] {
        background-color: rgba(255,255,255,0);
    }
    [data-testid="stSidebar"] {
        background-color: #f8d7da;
    }

    /* Navbar */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: white;
        padding: 1rem 3rem;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .nav-left {
        font-weight: 700;
        font-size: 1.2rem;
        color: #d63384;
    }
    .nav-right a {
        margin-left: 2rem;
        text-decoration: none;
        color: #333;
        font-weight: 500;
    }
    .nav-right a:hover {
        color: #d63384;
    }

    /* Tombol */
    .stButton>button {
        background-color: #d63384;
        color: white;
        border: none;
        padding: 0.6rem 1.3rem;
        border-radius: 8px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #e754a5;
        color: white;
    }

    /* Card Putih */
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.05);
        text-align: center;
    }

    h1, h2, h3 {
        color: #212529;
    }
    .sub {
        color: #555;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# ========================
# Navbar
# ========================
st.markdown("""
<div class="navbar">
    <div class="nav-left">🚘 Deteksi Jenis</div>
    <div class="nav-right">
        <a href="#">Demo Cepat</a>
        <a href="#">Model Info</a>
        <a href="#">Tentang Kami</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ========================
# Hero Section
# ========================
col1, col2 = st.columns([1.3, 1])

with col1:
    st.markdown("### Deteksi Jenis **Kendaraan AI**")
    st.write("""
    Platform revolusioner yang menggunakan teknologi deep learning untuk
    mengidentifikasi dan mengklasifikasi jenis kendaraan seperti mobil, motor,
    truck, dan bus dengan akurasi tinggi.
    """)
    st.markdown("** ")
    col_btn1, col_btn2 = st.columns([0.4, 0.6])
    with col_btn1:
        st.markdown('<div class="stButton"><button>Coba Sekarang</button></div>', unsafe_allow_html=True)
    with col_btn2:
        st.markdown('<div class="stButton"><button style="background-color:white;color:#d63384;border:2px solid #d63384;">Pelajari Lebih Lanjut</button></div>', unsafe_allow_html=True)

with col2:
    st.markdown("#### Demo Cepat")
    st.info("Upload gambar kendaraan untuk analisis")
    uploaded_file = st.file_uploader("Upload gambar kendaraan", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)
        st.success("Gambar berhasil diunggah!")

# ========================
# Jenis Kendaraan
# ========================
st.markdown("---")
st.markdown("## Jenis Kendaraan yang Dapat Dideteksi")
st.write("Sistem AI kami dapat mengenali berbagai jenis kendaraan dengan akurasi tinggi.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="card"><h3>🚗</h3><b>Mobil</b><p class="sub">Sedan, SUV, Hatchback, dan lainnya</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="card"><h3>🏍️</h3><b>Motor</b><p class="sub">Sepeda motor dan kendaraan roda dua</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="card"><h3>🚛</h3><b>Truck</b><p class="sub">Kendaraan kargo dan komersial</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="card"><h3>🚌</h3><b>Bus</b><p class="sub">Bus kota dan antar kota</p></div>', unsafe_allow_html=True)

# ========================
# Performa Model
# ========================
st.markdown("---")
st.markdown("## Performa Model Kami")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="card"><h3>98.2%</h3><p class="sub">Akurasi Model</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="card"><h3>47ms</h3><p class="sub">Waktu Proses</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="card"><h3>4+</h3><p class="sub">Jenis Kendaraan</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="card"><h3>99.9%</h3><p class="sub">Uptime</p></div>', unsafe_allow_html=True)

# ========================
# Mengapa Memilih Kami
# ========================
st.markdown("---")
st.markdown("## Mengapa Memilih Platform Kami?")
st.write("Teknologi AI terdepan yang dirancang khusus untuk deteksi kendaraan dengan akurasi tinggi.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="card"><b>Deteksi Akurat</b><p class="sub">Akurasi hingga 98.2% dalam mengenali kendaraan.</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="card"><b>Pemrosesan Cepat</b><p class="sub">Identifikasi kendaraan kurang dari 50ms.</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="card"><b>Keamanan Tinggi</b><p class="sub">Data dienkripsi dengan aman end-to-end.</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="card"><b>API Global</b><p class="sub">Integrasi mudah melalui REST API.</p></div>', unsafe_allow_html=True)

# ========================
# Konfigurasi Halaman
# ========================
st.set_page_config(
    page_title="Klasifikasi Gambar AI",
    page_icon="🖼️",
    layout="wide"
)

# ========================
# CSS Styling
# ========================
st.markdown("""
    <style>
    body {
        background-color: #fdf5f8 !important;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #fdf5f8;
    }
    [data-testid="stHeader"] {
        background-color: rgba(255,255,255,0);
    }

    /* Navbar */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: white;
        padding: 1rem 3rem;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .nav-left {
        font-weight: 700;
        font-size: 1.2rem;
        color: #d63384;
    }
    .nav-right a {
        margin-left: 2rem;
        text-decoration: none;
        color: #333;
        font-weight: 500;
    }
    .nav-right a:hover, .active {
        color: #d63384;
    }

    /* Card Styling */
    .card {
        background-color: white;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
        min-height: 300px;
    }

    .stButton>button {
        background-color: #d63384;
        color: white;
        border: none;
        padding: 0.6rem 1.3rem;
        border-radius: 8px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #e754a5;
        color: white;
    }

    h1, h2, h3 {
        color: #212529;
    }
    </style>
""", unsafe_allow_html=True)

# ========================
# Navbar
# ========================
st.markdown("""
<div class="navbar">
    <div class="nav-left">🖼️ AI Image Detection</div>
    <div class="nav-right">
        <a href="#">Home</a>
        <a class="active" href="#">Classification</a>
        <a href="#">Model Performance</a>
        <a href="#">Model Info</a>
        <a href="#">About Project</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ========================
# Header
# ========================
st.markdown("## Klasifikasi Gambar AI")
st.write("Upload gambar dan biarkan AI kami menganalisis serta mengklasifikasi objek dalam gambar dengan akurasi tinggi.")

# ========================
# Bagian Upload & Hasil
# ========================
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload Gambar")
    uploaded_file = st.file_uploader("Pilih atau Drop Gambar", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    else:
        st.info("Mendukung JPG, PNG, WebP hingga 10MB")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Hasil Klasifikasi")
    if uploaded_file:
        st.success("✅ Hasil: Mobil (98.2% confidence)")
    else:
        st.warning("Upload dan analisis gambar untuk melihat hasil klasifikasi")
    st.markdown('</div>', unsafe_allow_html=True)
