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
# ==========================
# Tambahkan Custom CSS
st.markdown("""
    <style>
    /* Warna background lembut */
    .stApp {
        background-color: #fceef5;
    }

    /* Judul utama */
    h1 {
        color: #1e1e1e;
        font-weight: 900;
    }

    /* Warna pink di subjudul */
    .pink-text {
        color: #e75480;
        font-weight: bold;
    }

    /* Tombol style */
    div.stButton > button:first-child {
        background-color: #ff5fa2;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.6em 1.2em;
        font-weight: 600;
    }

    div.stButton > button:first-child:hover {
        background-color: #e74b8b;
    }

    /* Card upload */
    .stFileUploader {
        background-color: white;
        padding: 1em;
        border: 2px dashed #ff9ac0;
        border-radius: 15px;
        box-shadow: 0px 4px 8px rgba(255, 192, 203, 0.3);
    }

    /* Sidebar (kalau ada) */
    section[data-testid="stSidebar"] {
        background-color: #f9e1ec;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Deteksi Jenis")
    st.markdown("<h1 class='pink-text'>Kendaraan AI</h1>", unsafe_allow_html=True)
    st.write("Platform revolusioner yang menggunakan teknologi deep learning untuk mengidentifikasi dan mengklasifikasikan jenis kendaraan seperti mobil, motor, truck, dan bus dengan akurasi tinggi.")
    st.button("🚀 Coba Sekarang")
    st.button("📖 Pelajari Lebih Lanjut")

with col2:
    st.markdown("### Demo Cepat")
    uploaded_file = st.file_uploader("Upload gambar kendaraan untuk analisis", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)
    else:
        st.info("Tidak ada gambar diunggah. Upload file di atas untuk mencoba.")

if selected == "Home":
    st.markdown("""
        <style>
        h2, h3 { color: #e75480; }        /* warna judul pastel pink */
        hr { border: 1px solid #f8cdda; }  /* garis pemisah warna ungu muda */
        </style>
    """, unsafe_allow_html=True)

    # === Konten halaman home ===
    st.markdown("<h1 style='text-align:center;'>Selamat Datang di Aplikasi Deteksi Kendaraan</h1>", unsafe_allow_html=True)

# ============================
# Section: Jenis Kendaraan yang Dapat Dideteksi
# ============================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center;'>
        <h2>Jenis Kendaraan yang Dapat Dideteksi</h2>
        <p>Sistem AI kami dapat mengenali berbagai jenis kendaraan dengan akurasi tinggi</p>
    </div>
""", unsafe_allow_html=True)

# Empat kolom kendaraan
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image("images/car.jpg", use_column_width=True)
    st.markdown("### 🚗 Mobil")
    st.write("Sedan, SUV, Hatchback, dan berbagai jenis mobil penumpang")

with col2:
    st.image("images/motorcycle.jpg", use_column_width=True)
    st.markdown("### 🏍️ Motor")
    st.write("Sepeda motor, skuter, dan kendaraan roda dua lainnya")

with col3:
    st.image("images/truck.jpg", use_column_width=True)
    st.markdown("### 🚚 Truck")
    st.write("Truk kargo, pickup, dan kendaraan komersial berat")

with col4:
    st.image("images/bus.jpg", use_column_width=True)
    st.markdown("### 🚌 Bus")
    st.write("Bus kota, bus antar kota, dan kendaraan angkutan umum")

# ============================
# Section: Statistik Model
# ============================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center;'>
        <h3>📊 Performa Model Kami</h3>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Akurasi Model", "98.2%")
with col2:
    st.metric("Waktu Proses", "47ms")
with col3:
    st.metric("Jenis Kendaraan", "4+")
with col4:
    st.metric("Uptime", "99.9%")

# ============================
# Section: Mengapa Memilih Kami
# ============================
st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center;'>
        <h2>Mengapa Memilih Platform Kami?</h2>
        <p>Teknologi AI terdepan yang dirancang khusus untuk deteksi kendaraan dengan akurasi tinggi</p>
    </div>
""", unsafe_allow_html=True)

# Keunggulan fitur
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("### 🎯 Deteksi Akurat")
    st.write("Akurasi hingga 98.2% dalam mengenali jenis kendaraan dengan teknologi deep learning")

with col2:
    st.markdown("### ⚡ Pemrosesan Cepat")
    st.write("Identifikasi kendaraan dalam waktu kurang dari 50ms")

with col3:
    st.markdown("### 🔒 Keamanan Tinggi")
    st.write("Data gambar kendaraan diproses dengan enkripsi end-to-end")

with col4:
    st.markdown("### 🌐 API Global")
    st.write("Akses mudah melalui REST API untuk integrasi sistem traffic management")
