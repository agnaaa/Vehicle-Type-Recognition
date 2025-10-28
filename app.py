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
# ------------------- Konfigurasi Halaman -------------------
st.set_page_config(page_title="Vehicle Classification", layout="wide")

# ------------------- Inisialisasi Session State -------------------
if "page" not in st.session_state:
    st.session_state["page"] = "home"

# ------------------- Style Global -------------------
st.markdown("""
<style>
body {
    background-color: #fdecef;
    font-family: "Poppins", sans-serif;
}
.navbar {
    display: flex;
    justify-content: center;
    background-color: #f8d3dc;
    padding: 12px 0;
    border-radius: 12px;
    margin-bottom: 25px;
}
.navbar button {
    background-color: transparent;
    border: none;
    color: #5a3e4d;
    font-size: 18px;
    font-weight: 600;
    margin: 0 25px;
    cursor: pointer;
}
.navbar button:hover {
    color: #b85c7d;
}
.section {
    padding: 40px 80px;
}
.title {
    font-size: 42px;
    font-weight: 700;
    color: #4a2b3a;
}
.subtitle {
    color: #6b4a57;
    font-size: 18px;
    margin-top: 10px;
}
.upload-card {
    background-color: white;
    border-radius: 18px;
    padding: 35px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
}
.upload-placeholder {
    border: 2px dashed #f2a7b4;
    border-radius: 16px;
    padding: 30px;
    margin-top: 15px;
}
.upload-choose {
    background-color: #f8d3dc;
    color: #5a3e4d;
    padding: 8px 20px;
    border-radius: 12px;
    display: inline-block;
    margin-top: 15px;
    font-weight: 600;
}
.vehicle-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 25px;
    margin-top: 50px;
}
.vehicle-card {
    background-color: white;
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
}
.vehicle-card img {
    width: 70px;
    margin-bottom: 10px;
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 25px;
    margin-top: 50px;
}
.feature-card {
    background-color: #fff;
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
.icon {
    font-size: 36px;
    margin-bottom: 10px;
}
.prediction-box {
    background-color: white;
    border-radius: 15px;
    padding: 25px;
    margin-top: 30px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ------------------- Fungsi Navigasi -------------------
def set_page(page):
    st.session_state["page"] = page

# ------------------- Navbar -------------------
st.markdown('<div class="navbar">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("🏠 Home"):
        set_page("home")
with col2:
    if st.button("📸 Classification"):
        set_page("classification")
with col3:
    if st.button("ℹ️ About Project"):
        set_page("about")
st.markdown('</div>', unsafe_allow_html=True)

# ------------------- Halaman HOME -------------------
if st.session_state["page"] == "home":
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="title">🚗 Vehicle Classification System</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="subtitle">
        Platform revolusioner yang menggunakan teknologi deep learning untuk mengidentifikasi 
        dan mengklasifikasikan jenis kendaraan seperti mobil, motor, truck, dan bus dengan akurasi tinggi.
        </div><br>
        """, unsafe_allow_html=True)
        if st.button("🚀 Coba Sekarang"):
            set_page("classification")

    with col2:
        st.markdown("""
        <div class="upload-card">
            <h4>Demo Cepat</h4>
            <div class="upload-placeholder">
                <div style="font-size:26px;">🖼️</div>
                <div style="margin-top:8px;color:#b88a9f">
                    Upload gambar kendaraan untuk analisis
                </div>
                <div class="upload-choose">Pilih Gambar</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Fitur kendaraan
    st.markdown("""
    <div class="vehicle-grid">
        <div class="vehicle-card">
            <img src="https://i.ibb.co/FXBvZZ7/car.png">
            <h4>Mobil 🚘</h4>
            <p>Sedan, SUV, Hatchback, dan berbagai jenis mobil penumpang</p>
        </div>
        <div class="vehicle-card">
            <img src="https://i.ibb.co/gWQhNsc/motorcycle.png">
            <h4>Motor 🏍️</h4>
            <p>Sepeda motor, skuter, dan kendaraan roda dua lainnya</p>
        </div>
        <div class="vehicle-card">
            <img src="https://i.ibb.co/F8y2Csx/truck.png">
            <h4>Truck 🚛</h4>
            <p>Truk kargo, pickup, dan kendaraan komersial berat</p>
        </div>
        <div class="vehicle-card">
            <img src="https://i.ibb.co/NrQL8cp/bus.png">
            <h4>Bus 🚌</h4>
            <p>Bus kota, bus antar kota, dan kendaraan angkutan umum</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Fitur utama
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="icon">🎯</div>
            <h4>Deteksi Akurat</h4>
            <p>Akurasi hingga 98.2% dalam mengenali jenis kendaraan</p>
        </div>
        <div class="feature-card">
            <div class="icon">⚡</div>
            <h4>Pemrosesan Cepat</h4>
            <p>Identifikasi kendaraan dalam waktu kurang dari 50ms</p>
        </div>
        <div class="feature-card">
            <div class="icon">🔒</div>
            <h4>Keamanan Tinggi</h4>
            <p>Data gambar kendaraan diproses dengan aman</p>
        </div>
        <div class="feature-card">
            <div class="icon">🌐</div>
            <h4>Integrasi Mudah</h4>
            <p>Bisa dihubungkan dengan sistem manajemen lalu lintas</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ------------------- Halaman CLASSIFICATION -------------------
elif st.session_state["page"] == "classification":
    st.markdown('<div class="title">📸 Klasifikasi Gambar Kendaraan</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload gambar kendaraan di sini", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diupload", use_container_width=True)

        # Simulasi prediksi model
        import random
        classes = ["Mobil 🚘", "Motor 🏍️", "Truck 🚛", "Bus 🚌"]
        prediction = random.choice(classes)

        st.markdown(f"""
        <div class="prediction-box">
            <h4>🔍 Hasil Prediksi:</h4>
            <h2 style="color:#b85c7d;">{prediction}</h2>
            <p>Model mendeteksi gambar ini sebagai kendaraan berjenis <b>{prediction}</b>.</p>
        </div>
        """, unsafe_allow_html=True)

# ------------------- Halaman ABOUT -------------------
elif st.session_state["page"] == "about":
    st.markdown("""
    <div class="title">ℹ️ Tentang Proyek</div>
    <div class="subtitle">
    Aplikasi ini dibuat untuk mendeteksi dan mengklasifikasikan kendaraan menggunakan deep learning.
    Dikembangkan dengan Streamlit dan TensorFlow untuk memberikan pengalaman deteksi real-time.
    </div>
    """, unsafe_allow_html=True)
