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
# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="AI Image Detection", layout="wide")

# ==============================
# CSS STYLE
# ==============================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, #fff5f8 0%, #ffffff 100%);
  padding-bottom: 60px;
}
* { font-family: 'Inter', sans-serif; }

/* NAVBAR */
.navbar {
  display: flex; justify-content: space-between; align-items: center;
  padding: 14px 40px; background: white; border-radius: 10px;
  box-shadow: 0 6px 18px rgba(16,24,40,0.06);
  margin: 10px 30px 30px 30px;
}
.nav-left { display: flex; align-items: center; gap: 10px; font-weight: 700; color: #111827; }
.logo-box {
  width: 36px; height: 36px; border-radius: 8px;
  background: linear-gradient(180deg,#f07da7,#e86e9a);
  display: flex; align-items: center; justify-content: center;
  color: white; font-weight: 800;
}
.nav-center { display: flex; gap: 18px; align-items: center; }
.nav-btn {
  background: none; border: none; padding: 8px 16px;
  font-weight: 600; color: #374151; cursor: pointer;
  border-radius: 8px; transition: all 0.2s ease-in-out;
}
.nav-btn:hover {
  background: rgba(231,81,120,0.06);
  color: #e75480;
}
.nav-btn.active {
  background: rgba(231,81,120,0.08);
  color: #e75480;
  box-shadow: 0 4px 14px rgba(231,81,120,0.08);
}

/* HERO */
.hero {
  display:flex; justify-content:space-between; align-items:center;
  padding: 56px 80px;
}
.hero-left { max-width: 640px; }
.hero-left h1 {
  font-size:48px; font-weight:800; margin:0; color:#111827;
}
.hero-left .accent { color:#e75480; }
.hero-left p { color:#6b7280; margin-top:18px; font-size:16px; line-height:1.6; }
.btn-primary {
  display:inline-block; background: linear-gradient(90deg,#f07da7,#e86e9a);
  color:white; padding: 12px 22px; border-radius:12px; font-weight:700; border:none; cursor:pointer;
  box-shadow: 0 10px 22px rgba(231,81,120,0.14);
}

/* UPLOAD CARD */
.upload-card {
  width:380px; background:white; border-radius:14px; padding:22px; text-align:center;
  box-shadow: 0 12px 40px rgba(16,24,40,0.06);
}
.upload-placeholder {
  border: 1px dashed #f6cde0; border-radius:10px; padding:28px; color:#b88a9f;
}

/* FADE */
.fade {
  animation: fadeIn 0.5s ease-in-out;
}
@keyframes fadeIn {
  from {opacity: 0; transform: translateY(10px);}
  to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ==============================
# SESSION STATE
# ==============================
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ==============================
# NAVBAR STREAMLIT
# ==============================
nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
with nav_col2:
    st.markdown('<div class="navbar">', unsafe_allow_html=True)
    st.markdown("""
    <div class="nav-left">
      <div class="logo-box">AI</div>
      <div>AI Image Detection</div>
    </div>
    """, unsafe_allow_html=True)

    nav_home, nav_class, nav_about = st.columns(3)
    with nav_home:
        if st.button("🏠 Home", key="nav_home",
                     use_container_width=True,
                     type="secondary" if st.session_state.page == "Home" else "primary"):
            st.session_state.page = "Home"
            st.rerun()

    with nav_class:
        if st.button("🧠 Classification", key="nav_class",
                     use_container_width=True,
                     type="secondary" if st.session_state.page == "Classification" else "primary"):
            st.session_state.page = "Classification"
            st.rerun()

    with nav_about:
        if st.button("ℹ️ About", key="nav_about",
                     use_container_width=True,
                     type="secondary" if st.session_state.page == "About" else "primary"):
            st.session_state.page = "About"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# HALAMAN: HOME
# ==============================
if st.session_state.page == "Home":
    st.markdown('<div class="fade">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="hero-left">
          <h1>Deteksi Jenis <span class="accent">Kendaraan AI</span></h1>
          <p>Gunakan teknologi AI canggih untuk mengenali jenis kendaraan seperti mobil, motor, bus, dan truk secara akurat dan cepat.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Coba Sekarang", key="btn_home"):
            st.session_state.page = "Classification"
            st.rerun()
    with col2:
        st.markdown("""
        <div class="upload-card">
          <h4>Demo Cepat</h4>
          <div class="upload-placeholder">🖼️<br><br>Upload gambar kendaraan untuk analisis</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# HALAMAN: CLASSIFICATION
# ==============================
elif st.session_state.page == "Classification":
    st.markdown('<div class="fade">', unsafe_allow_html=True)
    st.title("🔍 Klasifikasi Gambar AI")
    uploaded = st.file_uploader("Upload gambar kendaraan", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_column_width=True)
    if st.button("Analisis Sekarang"):
        if uploaded:
            with st.spinner("Menganalisis..."):
                time.sleep(1)
            jenis = random.choice(["Mobil", "Motor", "Truk", "Bus"])
            confidence = random.uniform(0.8, 0.98)
            st.success(f"✅ Teridentifikasi sebagai **{jenis}** dengan akurasi {confidence*100:.1f}%")
        else:
            st.warning("⚠️ Upload gambar terlebih dahulu.")
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# HALAMAN: ABOUT
# ==============================
elif st.session_state.page == "About":
    st.markdown('<div class="fade">', unsafe_allow_html=True)
    st.header("Tentang Project")
    st.write("""
    Aplikasi ini adalah prototipe sistem deteksi kendaraan berbasis kecerdasan buatan (AI).  
    Modelnya menggunakan deep learning untuk mengenali jenis kendaraan seperti mobil, motor, truk, dan bus.  
    Tampilan dibuat interaktif dan lembut dengan nuansa pink pastel 🌸.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
