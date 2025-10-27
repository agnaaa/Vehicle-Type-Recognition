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
# ----------------- Sidebar Menu -----------------
with st.sidebar:
    selected = option_menu(
        menu_title="Menu Utama",
        options=["Home", "Prediksi", "Tentang"],
        icons=["house", "camera", "info-circle"],
        default_index=0
    )

# ----------------- Halaman Home -----------------
if selected == "Home":
    st.markdown(
        "<h1 style='text-align: center; color: #333;'>🚗 Sistem Pengenalan Jenis Kendaraan</h1>",
        unsafe_allow_html=True
    )
    st.write(
        "Sistem ini menggunakan model AI untuk mengenali jenis kendaraan berdasarkan gambar yang diunggah."
    )

    st.markdown("### Jenis Kendaraan yang Dapat Dikenali")
    st.write("Sistem AI kami dapat mengenali berbagai jenis kendaraan berikut:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("images/car.jpg", use_container_width=True)
        st.markdown("**Mobil**")
        st.caption("Sedan, SUV, Hatchback, dan berbagai jenis mobil penumpang")

    with col2:
        st.image("images/motor.jpg", use_container_width=True)
        st.markdown("**Motor**")
        st.caption("Sepeda motor, skuter, dan kendaraan roda dua lainnya")

    with col3:
        st.image("images/truck.jpg", use_container_width=True)
        st.markdown("**Truk**")
        st.caption("Truk barang dan kendaraan niaga berat")

# ----------------- Halaman Prediksi -----------------
elif selected == "Prediksi":
    st.header("🔍 Prediksi Jenis Kendaraan")
    uploaded_file = st.file_uploader("Unggah gambar kendaraan:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)
        st.write("🚧 *Model prediksi akan ditambahkan di sini nanti.*")

# ----------------- Halaman Tentang -----------------
elif selected == "Tentang":
    st.header("Tentang Aplikasi")
    st.write("""
        Aplikasi ini dibuat untuk mendeteksi jenis kendaraan menggunakan model YOLO dan classifier CNN.
        Dikembangkan oleh tim Vehicle-Type-Recognition 2025.
    """)
