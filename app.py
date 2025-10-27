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
    yolo_model = YOLO("model/best.pt")

    try:
        # coba load dengan TensorFlow baru
        classifier = tf.keras.models.load_model("model/classifier_model.h5", safe_mode=False, compile=False)
    except Exception as e:
        st.error(f"Gagal memuat model klasifikasi: {e}")
        classifier = None

    return yolo_model, classifier

# ==========================
# UI
# Konfigurasi halaman
st.set_page_config(page_title="AI Image Detection", layout="wide")

# CSS untuk styling
st.markdown("""
    <style>
    body {
        background-color: #fdeef3;
    }
    .title {
        text-align: left;
        font-size: 48px;
        font-weight: 800;
        color: #ff4d88;
        margin-bottom: -10px;
    }
    .subtitle {
        font-size: 20px;
        color: #555;
        margin-bottom: 40px;
    }
    .button-primary {
        background-color: #ff4d88;
        color: white;
        padding: 12px 25px;
        border-radius: 12px;
        font-weight: 600;
        text-decoration: none;
    }
    .button-outline {
        border: 2px solid #ff4d88;
        color: #ff4d88;
        padding: 12px 25px;
        border-radius: 12px;
        text-decoration: none;
        font-weight: 600;
        margin-left: 10px;
    }
    .card {
        background-color: white;
        border-radius: 16px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        padding: 20px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header / Hero section
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("<div class='title'>Deteksi Jenis<br>Kendaraan AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Platform revolusioner yang menggunakan teknologi deep learning<br>untuk mengidentifikasi dan mengklasifikasi jenis kendaraan seperti mobil, motor, truk, dan bus.</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <a class="button-primary" href="#">🚗 Coba Sekarang</a>
        <a class="button-outline" href="#">📘 Pelajari Lebih Lanjut</a>
        """, unsafe_allow_html=True)
with col2:
    st.markdown("<div class='card'><h4>Demo Cepat</h4><p>Upload gambar kendaraan untuk analisis</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Pilih Gambar", type=["jpg", "png", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Jenis kendaraan
st.subheader("Jenis Kendaraan yang Dapat Dideteksi")
cols = st.columns(4)
kendaraan = ["Mobil", "Motor", "Truck", "Bus"]
deskripsi = ["Sedan, SUV, Hatchback", "Sepeda motor, skuter", "Truk kargo dan pickup", "Bus kota dan antar kota"]
for i in range(4):
    with cols[i]:
        st.markdown(f"<div class='card'><h4>{kendaraan[i]}</h4><p>{deskripsi[i]}</p></div>", unsafe_allow_html=True)

st.markdown("---")

# Statistik
st.subheader("Statistik Sistem")
cols2 = st.columns(4)
stat_label = ["Akurasi Model", "Waktu Proses", "Jenis Kendaraan", "Uptime"]
stat_value = ["98.2%", "47ms", "4+", "99.9%"]
for i in range(4):
    with cols2[i]:
        st.markdown(f"<div class='card'><h2>{stat_value[i]}</h2><p>{stat_label[i]}</p></div>", unsafe_allow_html=True)
