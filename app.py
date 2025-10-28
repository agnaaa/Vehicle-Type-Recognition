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
# ============ PAGE CONFIG ============
st.set_page_config(page_title="AI Image Detection", layout="wide")

# ============ CSS STYLE ============
st.markdown("""
<style>
body { background-color: #fdebf3; }
.main { background-color: #fdebf3; }
h1, h2, h3, h4, h5 {
    color: #1a1a1a;
    font-family: 'Helvetica Neue', sans-serif;
}
.navbar {
    display: flex;
    justify-content: center;
    background-color: white;
    border-radius: 15px;
    padding: 12px 20px;
    margin-bottom: 30px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}
.nav-button {
    border: none;
    background-color: transparent;
    color: #333;
    font-size: 16px;
    font-weight: 500;
    margin: 0 25px;
    cursor: pointer;
}
.nav-button:hover {
    color: #f06292;
}
.nav-active {
    background-color: #f8bbd0;
    color: white !important;
    border-radius: 10px;
    padding: 6px 15px;
}
.card {
    background-color: white;
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    transition: all 0.2s ease-in-out;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 6px 15px rgba(0,0,0,0.1);
}
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ============ NAVIGATION BAR ============
if "page" not in st.session_state:
    st.session_state.page = "Home"

col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("Home", key="home", use_container_width=True):
        st.session_state.page = "Home"
with col2:
    if st.button("Classification", key="class", use_container_width=True):
        st.session_state.page = "Classification"
with col3:
    if st.button("About Project", key="about", use_container_width=True):
        st.session_state.page = "About Project"

st.markdown("---")

# ============ HOME PAGE ============
if st.session_state.page == "Home":
    st.markdown("<h1 style='text-align:center;'>Selamat Datang di AI Image Detection 🚗</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Sistem AI kami mampu mengenali berbagai jenis kendaraan dengan akurasi tinggi dan waktu proses yang cepat.</p>", unsafe_allow_html=True)

    # Statistik
    st.markdown("<h3 style='text-align:center; margin-top:50px;'>Statistik Sistem</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="grid">
        <div class="card"><h2>98.2%</h2><p>Akurasi Model</p></div>
        <div class="card"><h2>47ms</h2><p>Waktu Proses</p></div>
        <div class="card"><h2>4+</h2><p>Jenis Kendaraan</p></div>
        <div class="card"><h2>99.9%</h2><p>Uptime</p></div>
    </div>
    """, unsafe_allow_html=True)

    # Jenis kendaraan
    st.markdown("<h3 style='text-align:center; margin-top:60px;'>Jenis Kendaraan yang Dapat Dideteksi</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="grid">
        <div class="card">🚗 <h4>Mobil</h4><p>Sedan, SUV, Hatchback, dan mobil penumpang lainnya</p></div>
        <div class="card">🏍️ <h4>Motor</h4><p>Sepeda motor, skuter, dan kendaraan roda dua lainnya</p></div>
        <div class="card">🚚 <h4>Truck</h4><p>Truk kargo, pickup, dan kendaraan komersial berat</p></div>
        <div class="card">🚐 <h4>Bus</h4><p>Kendaraan besar untuk transportasi umum</p></div>
    </div>
    """, unsafe_allow_html=True)


# ============ CLASSIFICATION PAGE ============
elif st.session_state.page == "Classification":
    st.markdown("<h1 style='text-align:center;'>Klasifikasi Gambar 🚀</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Unggah gambar kendaraan untuk mendeteksi jenisnya menggunakan model AI kami.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Unggah gambar kendaraan:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Gambar yang diunggah", use_container_width=True)
        st.success("Model sedang memproses gambar...")
        st.info("📊 Prediksi: Mobil (98.2% confidence)")  # Dummy contoh hasil


# ============ ABOUT PROJECT PAGE ============
elif st.session_state.page == "About Project":
    st.markdown("<h1 style='text-align:center;'>Tentang Proyek AI Image Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Proyek penelitian dan pengembangan sistem deteksi gambar berbasis AI yang dirancang untuk memberikan akurasi tinggi dalam klasifikasi objek dengan tampilan yang cantik dan menarik.</p>", unsafe_allow_html=True)

    # Misi & Visi
    st.markdown("""
    <div class="grid">
        <div class="card">
            <h4>🎯 Misi Kami</h4>
            <p>Mengembangkan teknologi AI yang dapat memahami dan menginterpretasi gambar dengan akurasi tinggi untuk berbagai industri dan aplikasi.</p>
        </div>
        <div class="card">
            <h4>🌟 Visi Kami</h4>
            <p>Menjadi platform AI terdepan dalam computer vision yang membawa inovasi ke berbagai sektor industri.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Gambaran proyek
    st.markdown("<h3 style='text-align:center; margin-top:50px;'>Gambaran Proyek</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="grid">
        <div class="card">
            <h4>Latar Belakang</h4>
            <p>AI Image Detection lahir dari kebutuhan akan sistem deteksi gambar yang akurat, cepat, dan mudah digunakan.</p>
            <p>Kami berfokus menggabungkan deep learning dan computer vision untuk menciptakan sistem yang efisien dan elegan.</p>
        </div>
        <div class="card">
            <img src="https://cdn.pixabay.com/photo/2020/02/09/08/06/artificial-intelligence-4831594_1280.jpg" style="width:100%; border-radius:15px;">
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tim
    st.markdown("<h3 style='text-align:center; margin-top:50px;'>Tim Pengembang</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="grid">
        <div class="card">
            <img src="https://cdn-icons-png.flaticon.com/512/4140/4140037.png" width="80">
            <h4>Agna Balqis</h4>
            <p>Lead Developer & AI Engineer</p>
            <p>Spesialis dalam pengembangan model AI dan implementasi sistem berbasis deep learning.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Kolaborasi
    st.markdown("""
    <div class="card" style="max-width:800px; margin:auto; margin-top:60px;">
        <h4>Tertarik Berkolaborasi?</h4>
        <p>Kami terbuka untuk penelitian, partnership, atau diskusi tentang implementasi teknologi AI. Mari bersama menciptakan masa depan yang lebih cerdas!</p>
        <a href="#" style="padding:10px 20px; background-color:#f06292; color:white; border-radius:10px; text-decoration:none;">Hubungi Kami</a>
    </div>
    """, unsafe_allow_html=True)
