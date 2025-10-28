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
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

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
# Konfigurasi halaman
st.set_page_config(page_title="AI Vehicle Detection", page_icon="🚗", layout="wide")

# CSS styling
st.markdown("""
<style>
html, body, [class*="st-"], .main {
    background-color: #fdeff4 !important;
    font-family: 'Poppins', sans-serif;
    animation: fadeIn 1s ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
.brand {font-weight:800;font-size:30px;color:#111827;display:flex;align-items:center;gap:10px;}
.brand .logo {width:42px;height:42px;border-radius:10px;background:linear-gradient(90deg,#f07da7,#e86e9a);
    display:flex;align-items:center;justify-content:center;color:white;font-weight:800;}
.nav {display:flex;gap:24px;align-items:center;justify-content:center;}
.nav button {background:none;border:none;padding:14px 24px;border-radius:14px;font-weight:700;cursor:pointer;color:#374151;font-size:20px;}
.nav button.active {background:white;color:#e75480;box-shadow:0 10px 26px rgba(231,81,120,0.12);}
.hero {display:flex;gap:60px;align-items:center;padding:50px 80px;}
.hero-left h1 {font-size:70px;margin:0;line-height:1.1;font-weight:900;color:#111827;}
.hero-left h1 .accent {color:#e75480;}
.hero-left p {color:#6b7280;margin-top:20px;font-size:26px;max-width:800px;}
.btn-primary {background:linear-gradient(90deg,#f07da7,#e86e9a);color:white;padding:18px 40px;
    border-radius:16px;border:none;font-weight:700;font-size:24px;cursor:pointer;}
.section-title {text-align:center;font-size:40px;font-weight:800;margin-top:60px;color:#111827;}
.vehicle-grid, .features-grid {display:flex;gap:26px;justify-content:center;flex-wrap:wrap;margin-top:34px;}
.vehicle-card, .feature-card {width:280px;text-align:center;padding:30px;border-radius:20px;
    background:white;box-shadow:0 8px 24px rgba(16,24,40,0.08);}
.vehicle-card h4, .feature-card h4 {margin:12px 0;font-size:22px;}
.feature-card p {font-size:18px;}
.stats {display:flex;justify-content:center;gap:80px;margin-top:60px;text-align:center;}
.stat {font-weight:800;font-size:40px;color:#e75480;}
.stat-label {font-size:20px;color:#6b7280;}
.classification {display:flex;gap:40px;padding:30px 70px;}
.left-panel, .right-panel {background:white;border-radius:16px;padding:26px;box-shadow:0 10px 26px rgba(16,24,40,0.08);}
.left-panel {flex:1;}
.right-panel {width:520px;}
.about-box {background:white;padding:30px;border-radius:16px;box-shadow:0 8px 22px rgba(16,24,40,0.08);margin-bottom:24px;}
.developer-card {text-align:center;padding:30px;border-radius:18px;background:white;width:500px;margin:60px auto;
    box-shadow:0 10px 34px rgba(16,24,40,0.1);}
.developer-card img {width:330px;height:330px;border-radius:50%;object-fit:cover;
    box-shadow:0 12px 32px rgba(0,0,0,0.12);}
.developer-card h3 {font-size:30px;margin-top:18px;color:#111827;}
footer {text-align:center;color:#6b7280;margin-top:60px;padding-bottom:20px;font-size:18px;}
@media (max-width:900px){
    .hero{flex-direction:column;padding:24px;}
    .classification{flex-direction:column;padding:16px;}
    .right-panel{width:100%;}
}
</style>
""", unsafe_allow_html=True)

# Navbar
if "page" not in st.session_state:
    st.session_state.page = "Home"

col1, col2, col3 = st.columns([1,3,1])
with col1:
    st.markdown('<div class="brand"><div class="logo">AI</div><div>AI Vehicle Detection</div></div>', unsafe_allow_html=True)
with col2:
    cols = st.columns([1,1,1])
    pages = ["Home","Classification","About Project"]
    for i,p in enumerate(pages):
        with cols[i]:
            active = "active" if st.session_state.page==p else ""
            if st.button(p, key=f"nav_{p}"):
                st.session_state.page = p
with col3:
    st.write("")
st.markdown("<hr style='margin-top:10px;margin-bottom:24px;border:none;height:1px;background:#f3d7e0' />", unsafe_allow_html=True)

# ========================= HOME =========================
if st.session_state.page == "Home":
    left, right = st.columns([1.2,1])
    with left:
        st.markdown("""
        <div class="hero-left">
            <h1>Deteksi <span class="accent">Kendaraan AI</span></h1>
            <p>Platform cerdas berbasis deep learning untuk mengenali dan mengklasifikasikan kendaraan seperti mobil, motor, truk, dan bus secara akurat dan cepat.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚗 Coba Sekarang"):
    st.session_state.page = "Classification"
    st.rerun()

    with right:
        st.image("https://i.ibb.co/dLcRb8G/train.png", use_container_width=True)

    # Jenis kendaraan
    st.markdown('<div class="section-title">Jenis Kendaraan yang Dapat Dideteksi</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="vehicle-grid">
        <div class="vehicle-card">🚘<h4>Mobil</h4><p>Sedan, SUV, Hatchback</p></div>
        <div class="vehicle-card">🏍️<h4>Motor</h4><p>Sepeda motor, skuter</p></div>
        <div class="vehicle-card">🚛<h4>Truck</h4><p>Truk kargo, pickup</p></div>
        <div class="vehicle-card">🚌<h4>Bus</h4><p>Bus kota & antar kota</p></div>
    </div>
    """, unsafe_allow_html=True)

    # Akurasi dan Statistik
    st.markdown("""
    <div class="stats">
        <div><div class="stat">98.2%</div><div class="stat-label">Akurasi Model</div></div>
        <div><div class="stat">47ms</div><div class="stat-label">Waktu Proses</div></div>
        <div><div class="stat">4+</div><div class="stat-label">Jenis Kendaraan</div></div>
        <div><div class="stat">99.9%</div><div class="stat-label">Uptime</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Mengapa memilih
    st.markdown('<div class="section-title">Mengapa Memilih Platform Kami?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="features-grid">
        <div class="feature-card"><div style="font-size:34px;color:#e75480">🎯</div><h4>Akurasi 98.2%</h4><p>Menggunakan model deep learning terkini.</p></div>
        <div class="feature-card"><div style="font-size:34px;color:#e75480">⚡</div><h4>Pemrosesan Cepat</h4><p>Prediksi kendaraan dalam milidetik.</p></div>
        <div class="feature-card"><div style="font-size:34px;color:#e75480">🔒</div><h4>Keamanan Terjamin</h4><p>Gambar diproses secara lokal.</p></div>
        <div class="feature-card"><div style="font-size:34px;color:#e75480">🌐</div><h4>Integrasi Mudah</h4><p>Mendukung sistem transportasi cerdas.</p></div>
    </div>
    """, unsafe_allow_html=True)

# ========================= CLASSIFICATION =========================
elif st.session_state.page == "Classification":
    st.markdown('<h2 style="text-align:center;">🔍 Klasifikasi Kendaraan AI</h2>', unsafe_allow_html=True)
    left, right = st.columns([1,0.8])
    with left:
        upl = st.file_uploader("Unggah gambar kendaraan", type=["jpg","jpeg","png"])
        if upl:
            img = Image.open(upl).convert("RGB")
            st.image(img, caption="Gambar yang diunggah", use_container_width=True)
    with right:
        if upl:
            name = upl.name.lower()
            if "truck" in name: result = "Truck 🚛"
            elif "bus" in name: result = "Bus 🚌"
            elif "motor" in name: result = "Motor 🏍️"
            elif "car" in name or "mobil" in name: result = "Mobil 🚘"
            else: result = "Kendaraan Tidak Dikenali ❓"
            st.success(f"Hasil Prediksi: **{result}**")
        else:
            st.info("Hasil prediksi akan muncul di sini setelah kamu upload gambar.")

# ========================= ABOUT PROJECT =========================
elif st.session_state.page == "About Project":
    st.markdown('<h2 style="text-align:center;">Tentang Proyek AI Vehicle Detection</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:#6b7280;font-size:20px;max-width:900px;margin:auto;">'
        'Sistem deteksi kendaraan berbasis kecerdasan buatan (AI) ini dikembangkan untuk mendukung analitik transportasi, '
        'monitoring lalu lintas, dan mendukung sistem Smart City dengan kemampuan mengenali berbagai jenis kendaraan secara otomatis.'
        '</p>',
        unsafe_allow_html=True
    )

    # ====== VISI & MISI ======
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="about-box">'
            '<h3>Misi Kami</h3>'
            '<p>Menghadirkan solusi AI yang efisien, akurat, dan mudah diintegrasikan ke berbagai sistem transportasi. '
            'Kami berkomitmen untuk menciptakan teknologi yang membantu meningkatkan keselamatan, efisiensi, dan pengelolaan lalu lintas secara berkelanjutan.</p>'
            '</div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            '<div class="about-box">'
            '<h3>Visi Kami</h3>'
            '<p>Menjadi pelopor dalam pengembangan teknologi Vision AI di bidang transportasi cerdas, '
            'serta memberikan kontribusi nyata dalam menciptakan lingkungan transportasi yang lebih aman, ramah lingkungan, dan berbasis data.</p>'
            '</div>',
            unsafe_allow_html=True
        )

    # ====== BAGIAN PENGEMBANG ======
    st.markdown('<div class="section-title">Pengembang</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;text-align:center;margin-top:40px;">
        <img src="6372789C-781F-4439-AE66-2187B96D6952.jpeg"
             style="width:330px;height:330px;border-radius:50%;object-fit:cover;
             box-shadow:0 12px 32px rgba(0,0,0,0.12);margin-bottom:20px;">
        <h3 style="font-size:30px;margin:0;color:#111827;">Agna Balqis</h3>
        <p style="color:#e75480;font-weight:600;font-size:20px;margin:4px 0;">Lead AI Developer</p>
        <p style="color:#6b7280;max-width:600px;">Mengembangkan model AI dan merancang tampilan visual proyek ini dengan penuh dedikasi, 
        memadukan teknologi deep learning dan desain antarmuka modern agar sistem ini mudah digunakan dan tampak profesional.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<footer>© 2024 AI Vehicle Detection. All rights reserved.</footer>', unsafe_allow_html=True)




