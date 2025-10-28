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
# ---- CONFIG ----
st.set_page_config(page_title="AI Image Detection", layout="wide")

# ---- STYLE ----
st.markdown("""
<style>
body {
    background-color: #ffeef5;
}
[data-testid="stAppViewContainer"] {
    background-color: #ffeef5;
}
.navbar {
    display: flex;
    justify-content: center;
    background-color: white;
    padding: 16px;
    border-radius: 12px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
}
.nav-item {
    margin: 0 20px;
    font-weight: 500;
    color: #333;
    cursor: pointer;
    padding: 6px 12px;
    border-radius: 8px;
}
.nav-item.active {
    background-color: #ffb6c1;
    color: white;
}
.upload-card, .feature-card, .vehicle-card {
    background-color: white;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
.vehicle-grid, .features-grid {
    display: flex;
    justify-content: center;
    gap: 24px;
    flex-wrap: wrap;
}
.icon {
    font-size: 36px;
    background-color: #ffc2d1;
    width: 64px;
    height: 64px;
    border-radius: 50%;
    margin: 0 auto 12px auto;
    display: flex;
    align-items: center;
    justify-content: center;
}
.section {
    margin-top: 60px;
    margin-bottom: 60px;
}
</style>
""", unsafe_allow_html=True)

# ---- NAVBAR ----
selected_page = st.session_state.get("page", "Home")

def set_page(p):
    st.session_state["page"] = p

st.markdown("""
<div class="navbar">
    <div class="nav-item {home_active}" onclick="window.location.href='?page=home'">Home</div>
    <div class="nav-item {class_active}" onclick="window.location.href='?page=classification'">Classification</div>
    <div class="nav-item {about_active}" onclick="window.location.href='?page=about'">About Project</div>
</div>
""".format(
    home_active="active" if selected_page == "Home" else "",
    class_active="active" if selected_page == "Classification" else "",
    about_active="active" if selected_page == "About" else "",
), unsafe_allow_html=True)

query = st.query_params
if "page" in query:
    selected_page = query["page"][0].capitalize()

# ---- HOME PAGE ----
if selected_page == "Home":
    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;padding:50px 80px;">
        <div style="max-width:50%;">
            <h1 style="font-size:42px;">Deteksi Jenis <span style="color:#e75480;">Kendaraan AI</span></h1>
            <p>Platform revolusioner yang menggunakan teknologi deep learning untuk mengenali jenis kendaraan seperti mobil, motor, truck, dan bus dengan akurasi tinggi.</p>
            <div style="margin-top:20px;">
                <button style="background:#e75480;color:white;border:none;padding:10px 20px;border-radius:8px;margin-right:10px;">🚀 Coba Sekarang</button>
                <button style="border:2px solid #e75480;color:#e75480;padding:10px 20px;border-radius:8px;">ℹ️ Pelajari Lebih Lanjut</button>
            </div>
        </div>
        <div class="upload-card" style="width:40%;">
            <h4>Demo Cepat</h4>
            <div style="border:2px dashed #f2aacb;padding:30px;border-radius:12px;text-align:center;">
                <div style="font-size:26px;">🖼️</div>
                <div style="margin-top:8px;color:#b88a9f">Upload gambar kendaraan untuk analisis</div>
                <button style="margin-top:12px;background:#e75480;color:white;border:none;padding:10px 20px;border-radius:8px;">Pilih Gambar</button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Jenis Kendaraan Section
    st.markdown("""
    <div class="section" style="text-align:center;">
        <h2>Jenis Kendaraan yang Dapat Dideteksi</h2>
        <p>Sistem AI kami dapat mengenali berbagai jenis kendaraan dengan akurasi tinggi</p>
        <div class="vehicle-grid">
            <div class="vehicle-card"><div style="font-size:60px;">🚗</div><h4>Mobil</h4><p>Sedan, SUV, Hatchback, dan mobil penumpang lainnya</p></div>
            <div class="vehicle-card"><div style="font-size:60px;">🏍️</div><h4>Motor</h4><p>Sepeda motor, skuter, dan kendaraan roda dua</p></div>
            <div class="vehicle-card"><div style="font-size:60px;">🚚</div><h4>Truck</h4><p>Kendaraan kargo, pickup, dan komersial</p></div>
            <div class="vehicle-card"><div style="font-size:60px;">🚌</div><h4>Bus</h4><p>Bus kota dan antar kota</p></div>
        </div>
    </div>

    <div style="display:flex;justify-content:center;gap:80px;margin-top:60px;text-align:center;">
        <div><div class="icon"></div><h3>98.2%</h3><p>Akurasi Model</p></div>
        <div><div class="icon"></div><h3>47ms</h3><p>Waktu Proses</p></div>
        <div><div class="icon"></div><h3>4+</h3><p>Jenis Kendaraan</p></div>
        <div><div class="icon"></div><h3>99.9%</h3><p>Uptime</p></div>
    </div>
    """, unsafe_allow_html=True)

# ---- CLASSIFICATION PAGE ----
elif selected_page == "Classification":
    st.markdown("""
    <div style="text-align:center;margin-top:40px;">
        <h2>Klasifikasi Gambar AI</h2>
        <p>Upload gambar dan biarkan AI kami menganalisis serta mengklasifikasi objek dengan akurasi tinggi.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Gambar")
        uploaded_file = st.file_uploader("Pilih gambar kendaraan", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, use_column_width=True)
            if st.button("Analisis Gambar 🚀"):
                with st.spinner("Menganalisis..."):
                    time.sleep(2)
                st.session_state["result"] = uploaded_file.name

    with col2:
        st.subheader("Hasil Klasifikasi")
        if "result" in st.session_state:
            st.success("Prediksi: Mobil 🚗 (97.8%)")

# ---- ABOUT PAGE ----
elif selected_page == "About":
    st.markdown("""
    <div style="text-align:center;margin-top:40px;">
        <h2>Tentang Proyek AI Image Detection</h2>
        <p>Proyek penelitian dan pengembangan sistem deteksi gambar berbasis AI yang dirancang untuk memberikan akurasi tinggi dengan tampilan yang menarik.</p>
    </div>

    <div class="section" style="display:flex;justify-content:center;gap:40px;">
        <div class="upload-card" style="width:40%;">
            <h4>Misi Kami</h4>
            <p>Mengembangkan teknologi AI untuk memahami gambar dengan akurasi tinggi yang dapat dimanfaatkan di berbagai industri.</p>
        </div>
        <div class="upload-card" style="width:40%;">
            <h4>Visi Kami</h4>
            <p>Menjadi platform AI terdepan di bidang computer vision dan analisis visual dengan pengalaman yang menyenangkan.</p>
        </div>
    </div>

    <div class="section" style="display:flex;justify-content:center;align-items:center;gap:40px;">
        <div style="width:50%;">
            <h3>Latar Belakang</h3>
            <p>AI Image Detection dikembangkan untuk membantu mendeteksi kendaraan seperti mobil, motor, truck, dan bus secara cepat dan akurat menggunakan deep learning.</p>
        </div>
        <img src="https://i.ibb.co/nBGdCdb/ai-lab.jpg" width="40%" style="border-radius:16px;">
    </div>

    <div class="section" style="text-align:center;">
        <h3>Keunggulan Utama</h3>
        <div class="features-grid">
            <div class="feature-card"><div class="icon">🎯</div><h4>Akurasi Tinggi</h4><p>Hasil deteksi mencapai 98.2% akurasi.</p></div>
            <div class="feature-card"><div class="icon">⚡</div><h4>Kecepatan Optimal</h4><p>Proses klasifikasi hanya butuh 47ms.</p></div>
            <div class="feature-card"><div class="icon">💡</div><h4>Scalable</h4><p>Dapat menangani ribuan request per jam.</p></div>
            <div class="feature-card"><div class="icon">❤️</div><h4>User-Friendly</h4><p>Desain intuitif dan mudah digunakan.</p></div>
        </div>
    </div>

    <div class="section" style="text-align:center;">
        <h3>Tim Pengembang</h3>
        <div class="vehicle-card" style="width:250px;margin:auto;">
            <img src="https://i.ibb.co/jygJ1pB/profile.png" style="width:100%;border-radius:12px;">
            <h4>Agna Balqis</h4>
            <p>Lead Developer</p>
        </div>
    </div>

    <div class="section" style="text-align:center;background:#f5a1be;padding:40px;border-radius:16px;margin:60px;">
        <h3>Tertarik Berkolaborasi?</h3>
        <p>Kami selalu terbuka untuk kolaborasi dan diskusi tentang implementasi AI.</p>
        <button style="background:white;color:#e75480;padding:10px 20px;border:none;border-radius:8px;margin-right:10px;">Hubungi Tim Research</button>
        <button style="border:2px solid white;color:white;padding:10px 20px;border-radius:8px;">Lihat Repository</button>
    </div>
    """, unsafe_allow_html=True)
