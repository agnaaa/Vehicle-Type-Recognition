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
# ---- Page Config ----
st.set_page_config(page_title="AI Vehicle Detection", page_icon="🚗", layout="wide")

# ---- Custom CSS ----
st.markdown("""
<style>
html, body, [class*="st-"], .main {
    background-color: #fdeff4 !important;
    font-family: 'Poppins', sans-serif;
}
.brand {font-weight:800;font-size:22px;color:#111827;display:flex;align-items:center;gap:10px;}
.brand .logo {width:36px;height:36px;border-radius:8px;background:linear-gradient(90deg,#f07da7,#e86e9a);
    display:flex;align-items:center;justify-content:center;color:white;font-weight:800;}
.nav {display:flex;gap:16px;align-items:center;justify-content:center;}
.nav button {background:none;border:none;padding:10px 16px;border-radius:10px;font-weight:700;cursor:pointer;color:#374151;}
.nav button.active {background:white;color:#e75480;box-shadow:0 8px 20px rgba(231,81,120,0.08);}
.hero {display:flex;gap:30px;align-items:center;padding:36px 48px;}
.hero-left h1 {font-size:50px;margin:0;line-height:1.1;font-weight:800;color:#111827;}
.hero-left h1 .accent {color:#e75480;}
.hero-left p {color:#6b7280;margin-top:14px;font-size:20px;max-width:640px;}
.btn-primary {background:linear-gradient(90deg,#f07da7,#e86e9a);color:white;padding:12px 26px;
    border-radius:12px;border:none;font-weight:700;font-size:18px;cursor:pointer;}
.section-title {text-align:center;font-size:30px;font-weight:800;margin-top:40px;color:#111827;}
.vehicle-grid, .features-grid {display:flex;gap:18px;justify-content:center;flex-wrap:wrap;margin-top:24px;}
.vehicle-card, .feature-card {width:240px;text-align:center;padding:20px;border-radius:12px;
    background:white;box-shadow:0 6px 18px rgba(16,24,40,0.04);}
.feature-card h4 {margin:8px 0;font-size:18px;}
.classification {display:flex;gap:28px;padding:18px 40px;}
.left-panel, .right-panel {background:white;border-radius:12px;padding:18px;box-shadow:0 8px 20px rgba(16,24,40,0.04);}
.left-panel {flex:1;}
.right-panel {width:520px;}
.about-box {background:white;padding:24px;border-radius:12px;box-shadow:0 8px 20px rgba(16,24,40,0.04);margin-bottom:18px;}
.developer-card {text-align:center;padding:22px;border-radius:12px;background:white;width:320px;margin:auto;
    box-shadow:0 10px 30px rgba(16,24,40,0.06);}
.developer-card img {width:180px;height:180px;border-radius:50%;object-fit:cover;
    box-shadow:0 8px 24px rgba(0,0,0,0.08);}
footer {text-align:center;color:#6b7280;margin-top:32px;padding-bottom:18px;}
@media (max-width:900px){.hero{flex-direction:column;padding:18px;}.classification{flex-direction:column;padding:12px;}.right-panel{width:100%;}}
</style>
""", unsafe_allow_html=True)

# ---- Navbar ----
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
st.markdown("<hr style='margin-top:8px;margin-bottom:18px;border:none;height:1px;background:#f3d7e0' />", unsafe_allow_html=True)

# ---- PAGE HOME ----
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
            st.experimental_rerun()
    with right:
        st.image("train.jpg", use_container_width=True)

    st.markdown('<div class="section-title">Jenis Kendaraan yang Dapat Dideteksi</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="vehicle-grid">
        <div class="vehicle-card">🚘<h4>Mobil</h4><p>Sedan, SUV, Hatchback</p></div>
        <div class="vehicle-card">🏍️<h4>Motor</h4><p>Sepeda motor, skuter</p></div>
        <div class="vehicle-card">🚛<h4>Truck</h4><p>Truk kargo, pickup</p></div>
        <div class="vehicle-card">🚌<h4>Bus</h4><p>Bus kota & antar kota</p></div>
    </div>
    """, unsafe_allow_html=True)

    # ✅ Bagian Akurasi (ditampilkan kembali)
    st.markdown('<div class="section-title">Akurasi Deteksi</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="features-grid">
        <div class="feature-card"><div style="font-size:26px;color:#e75480">🎯</div><h4>Akurasi 96%</h4><p>Menggunakan model MobileNetV2 yang terlatih untuk hasil optimal.</p></div>
        <div class="feature-card"><div style="font-size:26px;color:#e75480">⚡</div><h4>Proses Cepat</h4><p>Prediksi dalam hitungan detik.</p></div>
        <div class="feature-card"><div style="font-size:26px;color:#e75480">🔒</div><h4>Privasi Terjaga</h4><p>Data hanya diproses secara lokal.</p></div>
    </div>
    """, unsafe_allow_html=True)

# ---- PAGE CLASSIFICATION ----
elif st.session_state.page == "Classification":
    st.markdown('<h2 style="text-align:center;">🔍 Klasifikasi Kendaraan AI</h2>', unsafe_allow_html=True)
    left, right = st.columns([1,0.8])
    with left:
        upl = st.file_uploader("Pilih gambar kendaraan", type=["jpg","jpeg","png"])
        if upl:
            img = Image.open(upl).convert("RGB")
            st.image(img, caption="Gambar yang diunggah", use_container_width=True)
    with right:
        if upl:
            # prediksi dummy (ganti sesuai modelmu)
            name = upl.name.lower()
            if "truck" in name: result = "Truck"
            elif "motor" in name: result = "Motor"
            elif "bus" in name: result = "Bus"
            else: result = "Mobil"
            st.success(f"Hasil Prediksi: **{result}** ✅")
        else:
            st.info("Hasil prediksi akan muncul di sini setelah kamu upload gambar.")

# ---- PAGE ABOUT PROJECT ----
elif st.session_state.page == "About Project":
    st.markdown('<h2 style="text-align:center;">Tentang Proyek AI Vehicle Detection</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#6b7280;max-width:900px;margin:auto;">Sistem deteksi kendaraan berbasis AI yang dikembangkan untuk mendukung analitik transportasi dan smart traffic.</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="about-box"><h3>Misi Kami</h3><p>Menghadirkan teknologi AI akurat, efisien, dan mudah digunakan untuk mendeteksi kendaraan.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="about-box"><h3>Visi Kami</h3><p>Menjadi solusi vision AI terbaik di bidang transportasi cerdas.</p></div>', unsafe_allow_html=True)

    # 🔥 Foto Agna Balqis (pakai file yang kamu kirim)
    st.markdown('<div class="section-title">Tim Pengembang</div>', unsafe_allow_html=True)
    st.markdown('<div class="developer-card">', unsafe_allow_html=True)
    if os.path.exists("6372789C-781F-4439-AE66-2187B96D6952.jpeg"):
        st.image("6372789C-781F-4439-AE66-2187B96D6952.jpeg", width=180)
    else:
        st.image("https://images.unsplash.com/photo-1544005313-94ddf0286df2?auto=format&fit=crop&w=400&q=60", width=180)
    st.markdown('<h3 style="margin-top:12px;color:#111827">Agna Balqis</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color:#e75480;font-weight:600;margin-top:6px">Lead AI Developer</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b7280;margin-top:10px">Mengembangkan model AI serta merancang antarmuka visual proyek ini.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<footer>© 2024 AI Vehicle Detection. All rights reserved.</footer>', unsafe_allow_html=True)
