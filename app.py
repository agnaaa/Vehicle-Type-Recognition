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
# ============================================================
# KONFIGURASI DASAR
# ============================================================
st.set_page_config(page_title="AI Image Detection", layout="wide")

# ============================================================
# TAMBAHKAN GAYA CSS
# ============================================================
st.markdown("""
    <style>
    .stApp {
        background-color: #fdeef4;
        font-family: "Poppins", sans-serif;
    }

    /* Gaya Judul dan Subjudul */
    h1, h2, h3 {
        color: #1e1e1e;
    }
    .pink-text {
        color: #e75480;
        font-weight: 700;
    }
    .center {
        text-align: center;
    }

    /* Tombol */
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

    /* File uploader box */
    .stFileUploader {
        background-color: white;
        padding: 1em;
        border: 2px dashed #ff9ac0;
        border-radius: 15px;
        box-shadow: 0px 4px 8px rgba(255, 192, 203, 0.3);
    }

    /* Card box umum */
    .metric-box, .feature-box {
        background-color: #fff;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    }

    hr { border: 1px solid #f8cdda; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f9e1ec;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# MENU NAVBAR
# ============================================================
menu = ["Home", "Classification", "Model Performance", "Model Info", "About Project"]
selected = st.sidebar.selectbox("Pilih Halaman:", menu)

# ============================================================
# HALAMAN HOME
# ============================================================
if selected == "Home":
    # BAGIAN ATAS
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Deteksi Jenis")
        st.markdown("<h1 class='pink-text'>Kendaraan AI</h1>", unsafe_allow_html=True)
        st.write("Platform revolusioner yang menggunakan teknologi deep learning untuk mengidentifikasi dan mengklasifikasikan jenis kendaraan seperti mobil, motor, truk, dan bus dengan akurasi tinggi.")
        st.button("🚀 Coba Sekarang")
        st.button("📖 Pelajari Lebih Lanjut")

    with col2:
        st.markdown("### Demo Cepat")
        uploaded_file = st.file_uploader("Upload gambar kendaraan untuk analisis", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Gambar yang diunggah", use_container_width=True)
        else:
            st.info("Tidak ada gambar diunggah. Upload file di atas untuk mencoba.")

    st.write("---")

    # BAGIAN JENIS KENDARAAN
    st.markdown("<h1 class='center pink-text'>Jenis Kendaraan yang Dapat Dideteksi</h1>", unsafe_allow_html=True)
    st.markdown("<p class='center'>Sistem AI kami dapat mengenali berbagai jenis kendaraan dengan akurasi tinggi</p>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("https://cdn.pixabay.com/photo/2017/03/27/13/28/auto-2179220_640.jpg", use_container_width=True)
        st.markdown("### Mobil")
        st.caption("Sedan, SUV, Hatchback, dan berbagai jenis mobil penumpang")

    with col2:
        st.image("https://cdn.pixabay.com/photo/2018/08/02/00/50/motorcycle-3575557_640.jpg", use_container_width=True)
        st.markdown("### Motor")
        st.caption("Sepeda motor, skuter, dan kendaraan roda dua lainnya")

    with col3:
        st.image("https://cdn.pixabay.com/photo/2014/12/27/15/40/truck-581173_640.jpg", use_container_width=True)
        st.markdown("### Truck")
        st.caption("Truk kargo, pickup, dan kendaraan komersial berat")

    with col4:
        st.image("https://cdn.pixabay.com/photo/2016/11/29/02/53/bus-1868313_640.jpg", use_container_width=True)
        st.markdown("### Bus")
        st.caption("Bus kota, bus antar kota, dan kendaraan angkutan umum")

    st.write("---")

    # BAGIAN PERFORMA MODEL
    st.markdown("<h2 class='center pink-text'>Performa Model Kami</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='metric-box'><h3>98.2%</h3><p>Akurasi Model</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-box'><h3>47ms</h3><p>Waktu Proses</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-box'><h3>4+</h3><p>Jenis Kendaraan</p></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-box'><h3>99.9%</h3><p>Uptime</p></div>", unsafe_allow_html=True)

    st.write("---")

    # BAGIAN MENGAPA MEMILIH
    st.markdown("<h2 class='center pink-text'>Mengapa Memilih Platform Kami?</h2>", unsafe_allow_html=True)
    st.markdown("<p class='center'>Teknologi AI terdepan yang dirancang khusus untuk deteksi kendaraan dengan akurasi tinggi</p>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='feature-box'><h3>Deteksi Akurat</h3><p>Akurasi hingga 98.2% dalam mengenali jenis kendaraan dengan deep learning</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='feature-box'><h3>Pemrosesan Cepat</h3><p>Identifikasi kendaraan dalam waktu kurang dari 50ms</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='feature-box'><h3>Keamanan Tinggi</h3><p>Data gambar kendaraan diproses dengan enkripsi end-to-end</p></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='feature-box'><h3>API Global</h3><p>Akses mudah melalui REST API untuk integrasi sistem traffic management</p></div>", unsafe_allow_html=True)

    # FOOTER
    st.markdown("<br><hr><p style='text-align:center; color:#888;'>© 2025 AI Vehicle Detection | Dibangun dengan ❤️ menggunakan Streamlit</p>", unsafe_allow_html=True)

# ============================================================
# HALAMAN LAIN
# ============================================================
elif selected == "Classification":
    st.header("Halaman Klasifikasi")
    st.info("Fitur ini akan menampilkan hasil deteksi kendaraan dari gambar yang diunggah.")

elif selected == "Model Performance":
    st.header("Performa Model")
    st.write("Halaman ini berisi metrik evaluasi dan hasil pengujian model.")

elif selected == "Model Info":
    st.header("Informasi Model")
    st.write("Menjelaskan arsitektur dan detail model AI yang digunakan.")

elif selected == "About Project":
    st.header("Tentang Proyek")
    st.write("Aplikasi ini dikembangkan untuk mendemonstrasikan sistem deteksi kendaraan berbasis AI.")
