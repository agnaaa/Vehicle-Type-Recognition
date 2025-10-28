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
# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="AI Image Detection",
    layout="wide"
)

# =============================
# CUSTOM CSS
# =============================
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #fdeef2 0%, #ffffff 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Navbar */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 4rem;
        background-color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border-radius: 0 0 20px 20px;
        position: sticky;
        top: 0;
        z-index: 999;
    }
    .navbar-left {
        font-weight: 700;
        font-size: 20px;
        color: #1f2937;
    }
    .navbar-left span {
        color: #ec5c9a;
    }
    .navbar-right {
        display: flex;
        gap: 2rem;
        font-weight: 600;
    }
    .navbar-item {
        color: #1f2937;
        text-decoration: none;
        transition: 0.3s;
    }
    .navbar-item:hover {
        color: #ec5c9a;
    }
    .navbar-item.active {
        color: #ec5c9a;
        border-bottom: 3px solid #ec5c9a;
        padding-bottom: 4px;
    }

    /* Cards */
    .upload-card, .result-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(236,92,154,0.1);
        width: 100%;
        text-align: center;
    }
    .example-card {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 8px 25px rgba(236,92,154,0.1);
        text-align: center;
    }

    /* Floating button */
    .talk-btn {
        position: fixed;
        bottom: 25px;
        right: 25px;
        background-color: #ec5c9a;
        color: white;
        padding: 12px 24px;
        border-radius: 30px;
        font-weight: 600;
        text-decoration: none;
        box-shadow: 0 4px 15px rgba(236,92,154,0.4);
    }
    .talk-btn:hover { background-color: #e34c8f; }
    </style>
""", unsafe_allow_html=True)

# =============================
# NAVIGATION STATE
# =============================
if "page" not in st.session_state:
    st.session_state.page = "Home"

def switch_page(page):
    st.session_state.page = page

# =============================
# NAVBAR HTML
# =============================
st.markdown(f"""
<div class="navbar">
    <div class="navbar-left">AI <span>Image Detection</span></div>
    <div class="navbar-right">
        <a href="#" class="navbar-item {'active' if st.session_state.page == 'Home' else ''}" onClick="window.parent.postMessage({{type: 'streamlit:setSessionState', key: 'page', value: 'Home'}}, '*')">Home</a>
        <a href="#" class="navbar-item {'active' if st.session_state.page == 'Classification' else ''}" onClick="window.parent.postMessage({{type: 'streamlit:setSessionState', key: 'page', value: 'Classification'}}, '*')">Classification</a>
        <a href="#" class="navbar-item {'active' if st.session_state.page == 'About Project' else ''}" onClick="window.parent.postMessage({{type: 'streamlit:setSessionState', key: 'page', value: 'About Project'}}, '*')">About Project</a>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================
# HOME PAGE
# =============================
if st.session_state.page == "Home":
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
        <div style="margin-top:4rem;">
            <h1 style="font-size:48px; font-weight:800; color:#1f2937;">
                Deteksi Jenis <br><span style="color:#ec5c9a;">Kendaraan AI</span>
            </h1>
            <p style="font-size:16px; color:#6b7280; margin-top:1rem; line-height:1.6;">
            Platform revolusioner yang menggunakan teknologi <b>deep learning</b> 
            untuk mengidentifikasi dan mengklasifikasi jenis kendaraan seperti mobil, 
            motor, truk, dan bus dengan akurasi tinggi.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='upload-card'><h4>Demo Cepat</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload gambar kendaraan untuk analisis", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Gambar kendaraan diunggah", use_container_width=True)
            st.success("✅ Gambar berhasil diunggah (Demo).")
        else:
            st.info("Silakan upload gambar kendaraan (JPG/PNG) untuk demo deteksi.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("---")
    st.markdown("""
        <div style='text-align:center; margin-top:4rem;'>
            <h2 style="color:#1f2937;">Jenis Kendaraan yang Dapat Dideteksi</h2>
            <p style='color:#6b7280;'>Sistem AI kami dapat mengenali berbagai jenis kendaraan dengan akurasi tinggi</p>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.image("https://i.ibb.co/FXBvZZ7/car.png", caption="Mobil", use_container_width=True)
    col2.image("https://i.ibb.co/gWQhNsc/motorcycle.png", caption="Motor", use_container_width=True)
    col3.image("https://i.ibb.co/F8y2Csx/truck.png", caption="Truk", use_container_width=True)
    col4.image("https://i.ibb.co/NrQL8cp/bus.png", caption="Bus", use_container_width=True)

# =============================
# CLASSIFICATION PAGE
# =============================
elif st.session_state.page == "Classification":
    st.markdown("""
    <div style='text-align:center; margin-top:2rem;'>
        <h2 style="color:#1f2937;">Klasifikasi Gambar AI</h2>
        <p style='color:#6b7280;'>Upload gambar dan biarkan AI kami menganalisis serta mengklasifikasi objek dalam gambar.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='upload-card'><h4>Upload Gambar</h4>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Pilih atau Drop Gambar", type=["jpg", "jpeg", "png"], key="class_upload")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='result-card'><h4>Hasil Klasifikasi</h4>", unsafe_allow_html=True)
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Gambar yang diunggah", use_container_width=True)
            st.success("Contoh hasil: 🚗 Mobil (97.8%)")
        else:
            st.info("Upload dan analisis gambar untuk melihat hasil klasifikasi.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("---")
    st.markdown("<h4 style='text-align:center;'>Coba Gambar Contoh</h4>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.image("https://i.ibb.co/zZcdF12/cat.png", caption="Kucing")
    c2.image("https://i.ibb.co/k5XyN2H/dog.png", caption="Anjing")
    c3.image("https://i.ibb.co/FXBvZZ7/car.png", caption="Mobil")
    c4.image("https://i.ibb.co/4VvX8PW/apple.png", caption="Buah")

# =============================
# ABOUT PAGE
# =============================
elif st.session_state.page == "About Project":
    st.markdown("""
    <div style='text-align:center; margin-top:4rem;'>
        <h2 style="color:#1f2937;">Tentang Project Ini</h2>
        <p style='color:#6b7280; max-width:700px; margin:auto;'>
            Sistem ini dikembangkan menggunakan model deep learning untuk mendeteksi dan mengklasifikasi jenis kendaraan. 
            Dibangun menggunakan <b>TensorFlow</b> dan <b>Streamlit</b> sebagai antarmuka pengguna.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =============================
# FLOATING TALK BUTTON
# =============================
st.markdown("""<a class="talk-btn" href="#">💬 Talk with Us</a>""", unsafe_allow_html=True)
