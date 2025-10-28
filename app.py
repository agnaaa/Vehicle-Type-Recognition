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
import base64
from pathlib import Path


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
from ultralytics import YOLO
import webbrowser
import streamlit as st
from PIL import Image

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
from pathlib import Path

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
@keyframes fadeIn { from {opacity: 0;} to {opacity: 1;} }
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
            if st.button(p, key=f"nav_{p}", use_container_width=True):
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
    col_btn = st.columns([1,1,1])
    with col_btn[1]:
        if st.button("🚗 Coba Sekarang", use_container_width=True):
            st.session_state.page = "Classification"
            st.rerun()

    st.markdown('<div class="section-title">Jenis Kendaraan yang Dapat Dideteksi</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="vehicle-grid">
        <div class="vehicle-card">🚘<h4>Mobil</h4><p>Sedan, SUV, Hatchback</p></div>
        <div class="vehicle-card">🏍️<h4>Motor</h4><p>Sepeda motor, skuter</p></div>
        <div class="vehicle-card">🚛<h4>Truk</h4><p>Truk kargo, pickup</p></div>
        <div class="vehicle-card">🚌<h4>Bus</h4><p>Bus kota & antar kota</p></div>
    </div>
    """, unsafe_allow_html=True)

# ===========================================================
# ====================== CLASSIFICATION =====================
# ===========================================================
elif st.session_state.page == "Classification":
    st.markdown('<h2 style="text-align:center;">🔍 Klasifikasi Kendaraan AI</h2>', unsafe_allow_html=True)
    left, right = st.columns([1,0.8])

    # === Load model YOLO ===
    model_path = Path("model/best.pt")
    if model_path.exists():
        model = YOLO(str(model_path))
    else:
        st.error("❌ Gagal memuat model YOLO. Pastikan file 'model/best.pt' ada di folder 'model/'.")
        st.stop()

    with left:
        upl = st.file_uploader("Unggah gambar kendaraan", type=["jpg", "jpeg", "png"])
        if upl:
            img = Image.open(upl).convert("RGB")
            st.image(img, caption="Gambar yang diunggah", use_container_width=True)

    with right:
        if upl:
            with st.spinner("🔍 Mendeteksi kendaraan..."):
                results = model.predict(img, conf=0.5)
                result_img = results[0].plot()  # hasil dengan bounding box
                detected_classes = set([model.names[int(c)] for c in results[0].boxes.cls])

            if detected_classes:
                st.success(f"✅ Hasil Deteksi: {', '.join(detected_classes).title()}")
                st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
            else:
                st.warning("⚠️ Tidak ada kendaraan terdeteksi. Coba gambar lain.")
        else:
            st.info("📸 Silakan upload gambar kendaraan terlebih dahulu.")

# ===========================================================
# ====================== ABOUT PROJECT ======================
# ===========================================================
elif st.session_state.page == "About Project":
    st.markdown('<h2 style="text-align:center;">Tentang Proyek AI Vehicle Detection</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;color:#6b7280;font-size:20px;max-width:900px;margin:auto;">
        Sistem deteksi kendaraan berbasis AI ini dikembangkan untuk mendukung analitik transportasi, keamanan lalu lintas,
        dan sistem transportasi cerdas masa depan.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Pengembang</div>', unsafe_allow_html=True)
    try:
        img_agna = Image.open("6372789C-781F-4439-AE66-2187B96D6952.jpeg")
        st.image(img_agna, width=280, caption="Agna Balqis", use_container_width=False)
        wa_url = "https://wa.me/6289669727601"
        if st.button("💬 Tertarik Berkolaborasi? Hubungi Pengembang", use_container_width=True):
            st.markdown(f"<meta http-equiv='refresh' content='0; url={wa_url}'>", unsafe_allow_html=True)
    except:
        st.warning("⚠️ Foto pengembang tidak ditemukan. Pastikan file foto ada di folder yang sama.")
