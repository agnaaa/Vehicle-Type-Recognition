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
# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Vehicle Classification", layout="wide")

# --- CSS DESAIN PASTEL ---
st.markdown("""
<style>
body {
    background-color: #fce4ec;
    color: #4a4a4a;
    font-family: 'Poppins', sans-serif;
}
.navbar {
    display: flex;
    justify-content: center;
    background-color: #f8bbd0;
    padding: 15px 0;
    border-radius: 12px;
    margin-bottom: 20px;
}
.nav-item {
    margin: 0 30px;
    font-weight: 600;
    color: #4a4a4a;
    cursor: pointer;
}
.nav-item:hover {
    color: #880e4f;
}
.hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 40px;
    border-radius: 15px;
    background-color: #fce4ec;
}
.hero-left {
    width: 55%;
}
.hero-left h1 {
    color: #880e4f;
    font-size: 48px;
    margin-bottom: 20px;
}
.hero-left p {
    font-size: 18px;
    color: #5f5f5f;
    line-height: 1.6;
}
.upload-card {
    background-color: white;
    border-radius: 16px;
    padding: 30px;
    width: 350px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.vehicle-grid {
    display: flex;
    justify-content: space-between;
    margin-top: 50px;
}
.vehicle-card {
    background-color: white;
    border-radius: 16px;
    width: 23%;
    padding: 25px;
    text-align: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}
.vehicle-card img {
    width: 80px;
    margin-bottom: 10px;
}
.prediction-card {
    background-color: white;
    border-radius: 16px;
    padding: 30px;
    margin-top: 20px;
    text-align: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# --- NAVBAR INTERAKTIF ---
if "page" not in st.session_state:
    st.session_state.page = "Home"

cols = st.columns(3)
with cols[0]:
    if st.button("Home"):
        st.session_state.page = "Home"
with cols[1]:
    if st.button("Classification"):
        st.session_state.page = "Classification"
with cols[2]:
    if st.button("About Project"):
        st.session_state.page = "About"

# ================== HALAMAN HOME ==================
if st.session_state.page == "Home":
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
        <div class="hero-left">
            <h1>🚗 Vehicle Classification System</h1>
            <p>
            Sistem AI untuk mendeteksi jenis kendaraan secara otomatis seperti mobil, motor, truk, dan bus.
            <br><br>
            Unggah gambar kendaraanmu dan lihat hasil klasifikasinya secara instan!
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="upload-card">
            <h4>📤 Demo Cepat</h4>
            <p style="color:#b88a9f;">Upload gambar kendaraan untuk analisis</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Pilih gambar kendaraan", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, use_container_width=True)

    st.markdown("""
    <div class="vehicle-grid">
        <div class="vehicle-card">
            <img src="https://i.ibb.co/FXBvZZ7/car.png">
            <h4>🚙 Mobil</h4>
            <p>Sedan, SUV, Hatchback, dan mobil penumpang</p>
        </div>
        <div class="vehicle-card">
            <img src="https://i.ibb.co/gWQhNsc/motorcycle.png">
            <h4>🏍️ Motor</h4>
            <p>Sepeda motor dan skuter roda dua</p>
        </div>
        <div class="vehicle-card">
            <img src="https://i.ibb.co/F8y2Csx/truck.png">
            <h4>🚚 Truk</h4>
            <p>Truk barang, pickup, dan kendaraan berat</p>
        </div>
        <div class="vehicle-card">
            <img src="https://i.ibb.co/NrQL8cp/bus.png">
            <h4>🚌 Bus</h4>
            <p>Bus kota dan antar kota</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================== HALAMAN CLASSIFICATION ==================
elif st.session_state.page == "Classification":
    st.markdown("<h2 style='color:#880e4f;text-align:center;'>🔍 Klasifikasi Kendaraan</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload gambar kendaraan", type=["jpg","png","jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        # Dummy prediction
        labels = ["Mobil 🚗", "Motor 🏍️", "Truk 🚚", "Bus 🚌"]
        pred = random.choice(labels)
        st.markdown(f"""
        <div class="prediction-card">
            <h3>Hasil Prediksi:</h3>
            <h2 style='color:#ad1457'>{pred}</h2>
            <p>Tingkat kepercayaan: {round(random.uniform(90, 99.9),2)}%</p>
        </div>
        """, unsafe_allow_html=True)

# ================== HALAMAN ABOUT PROJECT ==================
elif st.session_state.page == "About":
    st.markdown("<h2 style='color:#880e4f;text-align:center;'>📘 Tentang Proyek</h2>", unsafe_allow_html=True)
    st.write("""
    Proyek ini bertujuan mengembangkan sistem klasifikasi kendaraan berbasis kecerdasan buatan (AI)
    untuk mengenali jenis kendaraan dari gambar.  
    Dibangun dengan Streamlit, sistem ini akan dihubungkan dengan model deep learning untuk prediksi nyata.
    """)
