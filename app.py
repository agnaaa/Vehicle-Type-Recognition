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

    # Nonaktifkan sementara classifier yang rusak
    classifier = None

    return yolo_model, classifier
# ==========================
# UI
# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="AI Image Detection",
    page_icon="🚗",
    layout="wide"
)

# ==========================
# CSS Custom Styling
# ==========================
st.markdown("""
    <style>
    body {
        background-color: #fff5f8;
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .title {
        font-size: 48px;
        font-weight: 800;
        color: #111827;
    }
    .highlight {
        color: #e86e9a;
    }
    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .button-primary {
        background: linear-gradient(90deg, #f07da7, #e86e9a);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 16px;
        cursor: pointer;
    }
    .button-outline {
        border: 2px solid #f07da7;
        background-color: transparent;
        color: #f07da7;
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 16px;
        cursor: pointer;
    }
    .demo-box {
        background: white;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        text-align: center;
    }
    .demo-title {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 20px;
    }
    .upload-text {
        color: #6b7280;
        font-size: 15px;
    }
    .footer {
        position: fixed;
        right: 30px;
        bottom: 20px;
        background: #e86e9a;
        color: white;
        padding: 12px 20px;
        border-radius: 30px;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# Header dan Navigasi
# ==========================
st.markdown(
    """
    <div style='display:flex; justify-content:space-between; align-items:center;'>
        <h3 style='color:#e86e9a; font-weight:700;'>🚗 AI Image Detection</h3>
        <div style='display:flex; gap:25px;'>
            <a href='#' style='color:#e86e9a; text-decoration:none; font-weight:600;'>Home</a>
            <a href='#' style='color:#6b7280; text-decoration:none;'>Classification</a>
            <a href='#' style='color:#6b7280; text-decoration:none;'>Model Performance</a>
            <a href='#' style='color:#6b7280; text-decoration:none;'>Model Info</a>
            <a href='#' style='color:#6b7280; text-decoration:none;'>About Project</a>
        </div>
    </div>
    <hr style='margin-top:10px;'>
    """,
    unsafe_allow_html=True
)

# ==========================
# Konten Utama
# ==========================
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("<div class='title'>Deteksi Jenis <span class='highlight'>Kendaraan AI</span></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Platform revolusioner yang menggunakan teknologi deep learning untuk "
        "mengidentifikasi dan mengklasifikasikan jenis kendaraan seperti mobil, motor, truck, dan bus dengan akurasi tinggi.</div>",
        unsafe_allow_html=True
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<button class='button-primary'>🚀 Coba Sekarang</button>", unsafe_allow_html=True)
    with c2:
        st.markdown("<button class='button-outline'>📘 Pelajari Lebih Lanjut</button>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='demo-box'>", unsafe_allow_html=True)
    st.markdown("<div class='demo-title'>Demo Cepat</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload gambar kendaraan untuk analisis", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar kendaraan diunggah", use_container_width=True)
        st.success("✅ Gambar berhasil diunggah! Model siap menganalisis.")
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# Footer
# ==========================
st.markdown("<div class='footer'>💬 Talk with Us</div>", unsafe_allow_html=True)


