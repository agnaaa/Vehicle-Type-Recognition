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

.navbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 40px; background: white; border-radius: 10px;
  box-shadow: 0 6px 18px rgba(16,24,40,0.06);
  margin: 10px 30px 30px 30px;
}
.nav-left { display:flex; align-items:center; gap:10px; font-weight:700; color:#111827; }
.logo-box {
  width:36px; height:36px; border-radius:8px;
  background: linear-gradient(180deg,#f07da7,#e86e9a);
  display:flex; align-items:center; justify-content:center;
  color:white; font-weight:800;
}
.nav-center { display:flex; gap:22px; align-items:center; }
.nav-link {
  padding:8px 14px; border-radius:8px; font-weight:600;
  color:#374151; cursor:pointer;
}
.nav-link.active {
  background: rgba(231,81,120,0.08);
  color:#e75480;
  box-shadow: 0 4px 14px rgba(231,81,120,0.08);
}
.fade {
  animation: fadeIn 0.5s ease-in-out;
}
@keyframes fadeIn {
  from {opacity: 0; transform: translateY(10px);}
  to {opacity: 1; transform: translateY(0);}
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

/* VEHICLE GRID */
.vehicle-grid {
  display:flex; gap:22px; justify-content:center; margin:48px 80px;
  flex-wrap:wrap;
}
.vehicle-card {
  width: 240px; background:white; border-radius:12px; padding:18px; text-align:center;
  box-shadow: 0 8px 20px rgba(16,24,40,0.04);
}
.vehicle-card img { width:100%; height:110px; object-fit:contain; border-radius:8px; margin-bottom:10px; }

/* CLASSIFICATION */
.classification {
  display:flex; gap:32px; padding:36px 80px;
}
.left-card, .right-card {
  background:white; border-radius:14px; padding:22px; box-shadow: 0 12px 40px rgba(16,24,40,0.06);
}
.left-card { flex:1; min-height:420px; }
.right-card { width:520px; }
.pred-row { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:14px; }
.progress-wrap { flex:1; height:12px; background:#f3f4f6; border-radius:8px; overflow:hidden; }
.progress-bar { height:100%; border-radius:8px; }

.info-box {
  background:#f3f6ff; padding:12px; border-radius:10px; color:#065f46; margin-top:18px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# SESSION STATE
# ==============================
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ==============================
# NAVBAR (klik = ubah session_state)
# ==============================
col1, col2, col3 = st.columns([1,2,1])
st.markdown(
    f"""
    <div class="navbar">
      <div class="nav-left">
        <div class="logo-box">AI</div>
        <div>AI Image Detection</div>
      </div>
      <div class="nav-center">
        <span class="nav-link {'active' if st.session_state.page=='Home' else ''}" onClick="window.parent.postMessage({{'page':'Home'}}, '*')">Home</span>
        <span class="nav-link {'active' if st.session_state.page=='Classification' else ''}" onClick="window.parent.postMessage({{'page':'Classification'}}, '*')">Classification</span>
        <span class="nav-link {'active' if st.session_state.page=='About' else ''}" onClick="window.parent.postMessage({{'page':'About'}}, '*')">About Project</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Listener JS buat update state tanpa reload
st.markdown("""
<script>
window.addEventListener('message', (event) => {
    if (event.data.page) {
        window.parent.postMessage({ type: 'streamlit:setComponentValue', key: 'nav_page', value: event.data.page }, '*');
    }
});
</script>
""", unsafe_allow_html=True)

# Streamlit hack agar bisa handle postMessage event
nav_page = st.session_state.page
if "nav_page" in st.session_state:
    nav_page = st.session_state.nav_page

# ==============================
# LOGIKA GANTI PAGE
# ==============================
if nav_page != st.session_state.page:
    st.session_state.page = nav_page

# ==============================
# PAGE: HOME
# ==============================
if st.session_state.page == "Home":
    st.markdown('<div class="fade">', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero">
      <div class="hero-left">
        <h1>Deteksi Jenis <span class="accent">Kendaraan AI</span></h1>
        <p>Platform revolusioner yang menggunakan teknologi deep learning untuk mengidentifikasi dan mengklasifikasikan jenis kendaraan seperti mobil, motor, truk, dan bus dengan akurasi tinggi.</p>
        """, unsafe_allow_html=True)
    if st.button("🚀 Coba Sekarang"):
        st.session_state.page = "Classification"
        st.rerun()
    st.markdown("""
      </div>
      <div class="upload-card">
        <h4>Demo Cepat</h4>
        <div class="upload-placeholder">🖼️<br><br>Upload gambar kendaraan untuk analisis</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# PAGE: CLASSIFICATION
# ==============================
elif st.session_state.page == "Classification":
    st.markdown('<div class="fade">', unsafe_allow_html=True)
    st.title("Klasifikasi Gambar AI")
    uploaded = st.file_uploader("Upload gambar kendaraan", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_column_width=True)
    if st.button("🔍 Analisis Gambar"):
        if uploaded:
            with st.spinner("Menganalisis..."):
                time.sleep(1)
            jenis = random.choice(["Mobil", "Motor", "Truk", "Bus"])
            confidence = random.uniform(0.8, 0.98)
            st.success(f"🚗 Teridentifikasi sebagai **{jenis}** dengan akurasi {confidence*100:.1f}%")
        else:
            st.warning("Silakan upload gambar terlebih dahulu.")
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# PAGE: ABOUT
# ==============================
elif st.session_state.page == "About":
    st.markdown('<div class="fade">', unsafe_allow_html=True)
    st.header("Tentang Project")
    st.write("Aplikasi ini adalah demo deteksi jenis kendaraan menggunakan AI. Prediksi saat ini masih dummy, namun UI sudah menyerupai sistem nyata.")
    st.markdown('</div>', unsafe_allow_html=True)
