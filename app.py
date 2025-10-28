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
# ---------------------------
# Konfigurasi halaman
# ---------------------------
st.set_page_config(page_title="AI Image Detection", layout="wide")

# ---------------------------
# CSS Styling
# ---------------------------
st.markdown("""
    <style>
        body {
            background-color: #fdeff5;
        }
        .main {
            background-color: #fdeff5;
        }
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .nav-links a {
            margin: 0 1rem;
            text-decoration: none;
            color: black;
            font-weight: 500;
        }
        .nav-links a.active {
            color: #ff4081;
            border-bottom: 2px solid #ff4081;
            padding-bottom: 4px;
        }
        .title-section {
            padding: 4rem 6rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .left-section {
            width: 50%;
        }
        .right-section {
            width: 40%;
            background-color: white;
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .card-section {
            background-color: white;
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .pink-btn {
            background-color: #ff4081;
            color: white;
            border-radius: 10px;
            padding: 0.7rem 1.5rem;
            font-weight: 600;
            border: none;
        }
        .outline-btn {
            border: 2px solid #ff4081;
            color: #ff4081;
            border-radius: 10px;
            padding: 0.7rem 1.5rem;
            font-weight: 600;
            background-color: transparent;
        }
        .footer-btn {
            position: fixed;
            bottom: 20px;
            right: 30px;
            background-color: #ff4081;
            color: white;
            border-radius: 50px;
            padding: 0.8rem 1.5rem;
            font-weight: 600;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .feature-card {
            background-color: white;
            border-radius: 15px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Navbar
# ---------------------------
st.markdown("""
<div class="navbar">
    <div class="logo">
        <h3 style="color:#333;">🧠 <b>AI Image Detection</b></h3>
    </div>
    <div class="nav-links">
        <a href="?page=home" class="{home_active}">Home</a>
        <a href="?page=classification" class="{class_active}">Classification</a>
        <a href="?page=about" class="{about_active}">About Project</a>
    </div>
</div>
""".format(
    home_active="active" if "page" not in st.query_params or st.query_params.get("page") == "home" else "",
    class_active="active" if st.query_params.get("page") == "classification" else "",
    about_active="active" if st.query_params.get("page") == "about" else ""
), unsafe_allow_html=True)

# ---------------------------
# Konten Halaman
# ---------------------------
page = st.query_params.get("page", "home")

# ======== HOME PAGE ========
if page == "home":
    st.markdown("""
        <div class="title-section">
            <div class="left-section">
                <h1 style="font-size: 40px; font-weight: 800;">Deteksi Jenis<br><span style="color:#ff4081;">Kendaraan AI</span></h1>
                <p style="font-size:18px; color:#444; margin-top:10px;">
                    Platform revolusioner yang menggunakan teknologi deep learning untuk mengidentifikasi 
                    dan mengklasifikasikan jenis kendaraan seperti mobil, motor, truck, dan bus dengan akurasi tinggi.
                </p>
                <div style="margin-top: 20px;">
                    <button class="pink-btn">🚀 Coba Sekarang</button>
                    <button class="outline-btn" style="margin-left:10px;">📘 Pelajari Lebih Lanjut</button>
                </div>
            </div>
            <div class="right-section">
                <h4>Demo Cepat</h4>
                <p>Upload gambar kendaraan untuk analisis</p>
                <input type="file" accept="image/*" style="margin-top:10px;">
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Bagian Jenis Kendaraan
    st.markdown("""
        <h2 style="text-align:center;">Jenis Kendaraan yang Dapat Dideteksi</h2>
        <p style="text-align:center;">Sistem AI kami dapat mengenali berbagai jenis kendaraan dengan akurasi tinggi</p>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("https://i.ibb.co/VVR8BdT/car.png", use_container_width=True)
        st.markdown("<h4 style='text-align:center;'>Mobil</h4><p style='text-align:center;'>Sedan, SUV, dan Hatchback</p>", unsafe_allow_html=True)
    with col2:
        st.image("https://i.ibb.co/qdsv07J/motor.png", use_container_width=True)
        st.markdown("<h4 style='text-align:center;'>Motor</h4><p style='text-align:center;'>Sepeda motor dan skuter</p>", unsafe_allow_html=True)
    with col3:
        st.image("https://i.ibb.co/Kw26rvS/truck.png", use_container_width=True)
        st.markdown("<h4 style='text-align:center;'>Truck</h4><p style='text-align:center;'>Truk kargo dan pickup</p>", unsafe_allow_html=True)
    with col4:
        st.image("https://i.ibb.co/f16XpRM/bus.png", use_container_width=True)
        st.markdown("<h4 style='text-align:center;'>Bus</h4><p style='text-align:center;'>Bus kota dan antar kota</p>", unsafe_allow_html=True)

    # Statistik
    st.markdown("""
        <div style="text-align:center; margin-top:2rem;">
            <div style="display:flex; justify-content:center; gap:50px;">
                <div><h3 style="color:#ff4081;">98.2%</h3><p>Akurasi Model</p></div>
                <div><h3 style="color:#ff4081;">47ms</h3><p>Waktu Proses</p></div>
                <div><h3 style="color:#ff4081;">4+</h3><p>Jenis Kendaraan</p></div>
                <div><h3 style="color:#ff4081;">99.9%</h3><p>Uptime</p></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Fitur tambahan
    st.markdown("""
        <h3 style="text-align:center; margin-top:3rem;">Mengapa Memilih Platform Kami?</h3>
        <p style="text-align:center;">Teknologi AI terdepan yang dirancang khusus untuk deteksi kendaraan dengan akurasi tinggi</p>
    """, unsafe_allow_html=True)
    feat1, feat2, feat3, feat4 = st.columns(4)
    feat1.markdown("<div class='feature-card'><h4>Deteksi Akurat</h4><p>Akurasi hingga 98.2%</p></div>", unsafe_allow_html=True)
    feat2.markdown("<div class='feature-card'><h4>Pemrosesan Cepat</h4><p>Analisis gambar <50ms</p></div>", unsafe_allow_html=True)
    feat3.markdown("<div class='feature-card'><h4>Keamanan Tinggi</h4><p>Enkripsi end-to-end</p></div>", unsafe_allow_html=True)
    feat4.markdown("<div class='feature-card'><h4>API Global</h4><p>Integrasi sistem mudah</p></div>", unsafe_allow_html=True)

# ======== CLASSIFICATION PAGE ========
elif page == "classification":
    st.markdown("""
        <h2 style="text-align:center;">Klasifikasi Gambar AI</h2>
        <p style="text-align:center;">Upload gambar dan biarkan AI menganalisis serta mengklasifikasikan objek dalam gambar</p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card-section'><h4>Upload Gambar</h4><p>Pilih atau Drop Gambar (JPG, PNG, WebP hingga 10MB)</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card-section'><h4>Hasil Klasifikasi</h4><p>Upload dan analisis gambar untuk melihat hasil</p></div>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align:center; margin-top:2rem;'>Coba Gambar Contoh</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.image("https://i.ibb.co/zNHTqMB/cat.png", caption="Kucing", use_container_width=True)
    c2.image("https://i.ibb.co/1zJX0zS/dog.png", caption="Anjing", use_container_width=True)
    c3.image("https://i.ibb.co/VVR8BdT/car.png", caption="Mobil", use_container_width=True)
    c4.image("https://i.ibb.co/yNcnDqH/apple.png", caption="Buah", use_container_width=True)

# ======== ABOUT PAGE ========
elif page == "about":
    st.title("Tentang Project")
    st.write("AI Image Detection adalah sistem deteksi kendaraan berbasis deep learning yang dirancang untuk mengidentifikasi berbagai jenis kendaraan dengan cepat dan akurat.")

# Tombol chat
st.markdown("""<button class="footer-btn">💬 Talk with Us</button>""", unsafe_allow_html=True)
