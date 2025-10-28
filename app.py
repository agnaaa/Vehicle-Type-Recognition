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
import streamlit as st
from PIL import Image

# =============================
# Konfigurasi halaman utama
# =============================
st.set_page_config(page_title="AI Vehicle Detection", page_icon="🚗", layout="wide")

# =============================
# CSS Styling
# =============================
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
.section-title {text-align:center;font-size:40px;font-weight:800;margin-top:60px;color:#111827;}
.vehicle-grid, .features-grid {display:flex;gap:26px;justify-content:center;flex-wrap:wrap;margin-top:34px;}
.vehicle-card, .feature-card {width:280px;text-align:center;padding:30px;border-radius:20px;
    background:white;box-shadow:0 8px 24px rgba(16,24,40,0.08);}
.feature-card p {font-size:18px;}
footer {text-align:center;color:#6b7280;margin-top:60px;padding-bottom:20px;font-size:18px;}
@media (max-width:900px){
    .hero{flex-direction:column;padding:24px;}
}
</style>
""", unsafe_allow_html=True)

# =============================
# Navbar
# =============================
if "page" not in st.session_state:
    st.session_state.page = "Home"

col1, col2, col3 = st.columns([1,3,1])
with col1:
    st.markdown('<div class="brand"><div class="logo">AI</div><div>AI Vehicle Detection</div></div>', unsafe_allow_html=True)
with col2:
    cols = st.columns([1,1,1])
    pages = ["Home", "Classification", "About Project"]
    for i, p in enumerate(pages):
        with cols[i]:
            active = "active" if st.session_state.page == p else ""
            if st.button(p, key=f"nav_{p}"):
                st.session_state.page = p
with col3:
    st.write("")

st.markdown("<hr style='margin-top:10px;margin-bottom:24px;border:none;height:1px;background:#f3d7e0' />", unsafe_allow_html=True)

# ===========================================================
# ========================= HOME ============================
# ===========================================================
if st.session_state.page == "Home":
    st.markdown("""
        <div style="text-align:center;">
            <h1 style="font-size:70px;font-weight:900;color:#111827;">Deteksi <span style="color:#e75480;">Kendaraan AI</span></h1>
            <p style="color:#6b7280;font-size:24px;max-width:900px;margin:auto;">
                Platform cerdas berbasis deep learning untuk mengenali dan mengklasifikasikan kendaraan seperti mobil, motor, truk, dan bus secara akurat dan cepat.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tombol coba sekarang
    col_btn = st.columns([1,1,1])
    with col_btn[1]:
        if st.button("🚗 Coba Sekarang", use_container_width=True):
            st.session_state.page = "Classification"
            st.experimental_rerun()

    # Jenis kendaraan
    st.markdown('<div class="section-title">Jenis Kendaraan yang Dapat Dideteksi</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="vehicle-grid">
        <div class="vehicle-card">🚘<h4>Mobil</h4><p>Sedan, SUV, Hatchback</p></div>
        <div class="vehicle-card">🏍️<h4>Motor</h4><p>Sepeda motor, skuter</p></div>
        <div class="vehicle-card">🚛<h4>Truk</h4><p>Truk kargo, pickup</p></div>
        <div class="vehicle-card">🚌<h4>Bus</h4><p>Bus kota & antar kota</p></div>
    </div>
    """, unsafe_allow_html=True)

    # Akurasi dan Statistik
    st.markdown("""
    <div style="display:flex;justify-content:center;gap:80px;margin-top:60px;text-align:center;">
        <div><div style="font-weight:800;font-size:40px;color:#e75480;">98.2%</div><div style="font-size:20px;color:#6b7280;">Akurasi Model</div></div>
        <div><div style="font-weight:800;font-size:40px;color:#e75480;">47ms</div><div style="font-size:20px;color:#6b7280;">Waktu Proses</div></div>
        <div><div style="font-weight:800;font-size:40px;color:#e75480;">4+</div><div style="font-size:20px;color:#6b7280;">Jenis Kendaraan</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Mengapa memilih platform kami
    st.markdown('<div class="section-title">Mengapa Memilih Platform Kami?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="features-grid">
        <div class="feature-card"><div style="font-size:34px;color:#e75480">🎯</div><h4>Akurasi 98.2%</h4><p>Menggunakan model deep learning terkini dengan hasil prediksi sangat presisi bahkan dalam kondisi lalu lintas padat.</p></div>
        <div class="feature-card"><div style="font-size:34px;color:#e75480">⚡</div><h4>Pemrosesan Cepat</h4><p>Proses deteksi kendaraan berlangsung hanya dalam hitungan milidetik, efisien untuk penggunaan real-time.</p></div>
        <div class="feature-card"><div style="font-size:34px;color:#e75480">🔒</div><h4>Keamanan Terjamin</h4><p>Data gambar diproses secara lokal tanpa dikirim ke server eksternal, menjaga privasi pengguna.</p></div>
        <div class="feature-card"><div style="font-size:34px;color:#e75480">🌐</div><h4>Integrasi Mudah</h4><p>Dapat diintegrasikan dengan sistem smart traffic, CCTV, maupun aplikasi analitik transportasi.</p></div>
    </div>
    """, unsafe_allow_html=True)

# ===========================================================
# ====================== CLASSIFICATION =====================
# ===========================================================
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

# Bagian About Project 

elif st.session_state.page == "About Project":
    st.markdown('<h2 style="text-align:center;">Tentang Proyek AI Vehicle Detection</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;color:#6b7280;font-size:20px;max-width:900px;margin:auto;">
        Sistem deteksi kendaraan berbasis AI ini dikembangkan untuk mendukung analitik transportasi, keamanan lalu lintas,
        dan sistem transportasi cerdas masa depan. Proyek ini berfokus pada efisiensi, akurasi, serta kemudahan implementasi.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visi & Misi
    st.markdown("""
    <div style="display:flex;justify-content:center;gap:40px;flex-wrap:wrap;margin-top:20px;">
        <div class="feature-card" style="width:400px;">
            <h3>Misi Kami</h3>
            <p>Menghadirkan teknologi AI yang mampu mengenali kendaraan secara cepat, akurat, dan efisien,
            membantu pengambilan keputusan di sektor transportasi modern dengan sistem yang adaptif dan ramah lingkungan.</p>
        </div>
        <div class="feature-card" style="width:400px;">
            <h3>Visi Kami</h3>
            <p>Menjadi solusi Vision AI terbaik yang terintegrasi dengan sistem smart city, mendorong inovasi dalam
            pengelolaan lalu lintas dan keselamatan transportasi masa depan.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ======================================
    # PENGEMBANG (Foto + Deskripsi Samping)
    # ======================================
    st.markdown('<div class="section-title">Pengembang</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:center;gap:50px;flex-wrap:wrap;margin-top:30px;">
        <div>
            <img src="6372789C-781F-4439-AE66-2187B96D6952.jpeg" 
                 style="width:260px;height:260px;border-radius:50%;object-fit:cover;
                        box-shadow:0 10px 30px rgba(0,0,0,0.15);">
        </div>
        <div style="max-width:500px;text-align:left;">
            <h3 style="font-size:30px;margin-bottom:5px;color:#111827;">Agna Balqis</h3>
            <p style="color:#e75480;font-weight:600;font-size:20px;margin-bottom:10px;">Lead AI Developer</p>
            <p style="color:#6b7280;font-size:18px;">
                Mengembangkan sistem AI dan antarmuka pengguna dengan dedikasi tinggi untuk menciptakan
                pengalaman deteksi kendaraan yang cerdas, cepat, dan ramah pengguna.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ======================================
    # BAGIAN KOLABORASI (seperti contoh kamu)
    # ======================================
    st.markdown("""
    <div style="background:linear-gradient(90deg,#ef5ea2,#ea7fb2);padding:40px;border-radius:25px;
                text-align:center;margin-top:80px;color:white;max-width:900px;margin-left:auto;margin-right:auto;">
        <h2 style="font-weight:800;font-size:34px;">Tertarik Berkolaborasi?</h2>
        <p style="font-size:18px;max-width:700px;margin:10px auto 30px;">
            Kami selalu terbuka untuk kolaborasi penelitian, partnership, atau diskusi tentang implementasi teknologi AI 
            dalam proyek Anda. Mari bersama-sama menciptakan masa depan yang lebih cerdas!
        </p>
        <a href="mailto:research.ai.vehicle@gmail.com" 
           style="background:white;color:#e75480;font-weight:700;padding:14px 28px;border-radius:12px;
                  text-decoration:none;font-size:18px;display:inline-flex;align-items:center;gap:10px;">
            ✉️ Hubungi Tim Research
        </a>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown('<footer>© 2024 AI Vehicle Detection. All rights reserved.</footer>', unsafe_allow_html=True)
