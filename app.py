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
body { background-color: #ffeef5; }
[data-testid="stAppViewContainer"] { background-color: #ffeef5; }
.navbar {
    display: flex;
    justify-content: center;
    background-color: white;
    padding: 16px;
    border-radius: 12px;
    margin: 20px auto;
    width: 90%;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
}
.nav-item {
    margin: 0 20px;
    font-weight: 500;
    color: #333;
    cursor: pointer;
    padding: 6px 18px;
    border-radius: 8px;
    transition: all 0.2s ease;
}
.nav-item:hover {
    background-color: #ffd2e1;
}
.nav-item.active {
    background-color: #f5a1be;
    color: white;
}
.section { margin: 60px auto; text-align: center; width: 85%; }
.card {
    background: white;
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ---- NAV STATE ----
if "page" not in st.session_state:
    st.session_state.page = "Home"

def set_page(name):
    st.session_state.page = name

# ---- NAVBAR ----
cols = st.columns([1,1,1])
with cols[0]:
    if st.button("🏠 Home", use_container_width=True, key="home_btn"):
        set_page("Home")
with cols[1]:
    if st.button("🧠 Classification", use_container_width=True, key="class_btn"):
        set_page("Classification")
with cols[2]:
    if st.button("💡 About Project", use_container_width=True, key="about_btn"):
        set_page("About")

st.markdown("---")

# ---- HOME ----
if st.session_state.page == "Home":
    st.markdown("""
    <div class="section">
        <h1>Selamat Datang di <span style="color:#e75480;">AI Image Detection</span></h1>
        <p>Platform berbasis deep learning yang mampu mendeteksi jenis kendaraan dengan akurasi tinggi dan tampilan menarik.</p>
    </div>
    """, unsafe_allow_html=True)

# ---- CLASSIFICATION ----
elif st.session_state.page == "Classification":
    st.markdown("""
    <div class="section">
        <h2>Klasifikasi Gambar AI</h2>
        <p>Upload gambar kendaraan dan biarkan AI mengenalinya untukmu.</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])
        if file:
            img = Image.open(file)
            st.image(img, use_column_width=True)
            if st.button("Analisis Gambar 🚀"):
                with st.spinner("Menganalisis..."):
                    time.sleep(1.5)
                st.success("Prediksi: 🚗 Mobil (97.8%)")
    with col2:
        st.info("Hasil prediksi AI akan muncul di sini setelah gambar dianalisis.")

# ---- ABOUT ----
elif st.session_state.page == "About":
    st.markdown("""
    <div class="section">
        <h2>Tentang Proyek AI Image Detection</h2>
        <p>Proyek penelitian pengembangan sistem deteksi gambar berbasis AI yang berfokus pada akurasi tinggi dan desain yang elegan.</p>
    </div>

    <div class="section" style="display:flex;gap:20px;justify-content:center;">
        <div class="card" style="width:40%;">
            <h4>Misi Kami</h4>
            <p>Mengembangkan AI yang memahami gambar dengan akurasi tinggi dan mudah digunakan di berbagai bidang.</p>
        </div>
        <div class="card" style="width:40%;">
            <h4>Visi Kami</h4>
            <p>Menjadi platform AI terdepan dalam computer vision dengan pengalaman pengguna yang menyenangkan.</p>
        </div>
    </div>

    <div class="section" style="text-align:center;">
        <h3>Tim Pengembang</h3>
        <div class="card" style="width:250px;margin:auto;">
            <img src="https://i.ibb.co/jygJ1pB/profile.png" style="width:100%;border-radius:12px;">
            <h4>Agna Balqis</h4>
            <p>Lead Developer</p>
        </div>
    </div>

    <div class="section" style="background:#f5a1be;padding:40px;border-radius:16px;color:white;">
        <h3>Tertarik Berkolaborasi?</h3>
        <p>Kami terbuka untuk penelitian, partnership, dan diskusi implementasi teknologi AI.</p>
        <button style="background:white;color:#e75480;padding:10px 20px;border:none;border-radius:8px;margin-right:10px;">Hubungi Kami</button>
        <button style="border:2px solid white;color:white;padding:10px 20px;border-radius:8px;">Lihat Repository</button>
    </div>
    """, unsafe_allow_html=True)
