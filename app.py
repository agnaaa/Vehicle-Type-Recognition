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
# ======================================
# Konfigurasi Halaman
# ======================================
st.set_page_config(page_title="Vehicle Type Recognition", page_icon="🚗", layout="wide")

# ======================================
# Sidebar Menu
# ======================================
with st.sidebar:
    selected = option_menu(
        menu_title="Menu Utama",
        options=["Home", "Prediksi", "Tentang"],
        icons=["house", "camera", "info-circle"],
        default_index=0
    )

# ======================================
# Halaman HOME
# ======================================
if selected == "Home":
    st.markdown(
        """
        <h1 style='text-align: center; color: #333;'>🚗 Sistem Pengenalan Jenis Kendaraan</h1>
        <p style='text-align: center; color: #666;'>
            Sistem ini menggunakan model AI untuk mengenali jenis kendaraan berdasarkan gambar yang diunggah.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### Jenis Kendaraan yang Dapat Dikenali")
    st.write("Sistem AI kami dapat mengenali berbagai jenis kendaraan berikut:")

    col1, col2, col3 = st.columns(3)

    with col1:
        if os.path.exists("images/car.jpg"):
            st.image("images/car.jpg", use_container_width=True)
        else:
            st.warning("⚠️ Gambar mobil tidak ditemukan. Pastikan file `images/car.jpg` ada di folder `images`.")
        st.markdown("**Mobil**")
        st.caption("Sedan, SUV, Hatchback, dan berbagai jenis mobil penumpang")

    with col2:
        if os.path.exists("images/motor.jpg"):
            st.image("images/motor.jpg", use_container_width=True)
        else:
            st.warning("⚠️ Gambar motor tidak ditemukan. Pastikan file `images/motor.jpg` ada di folder `images`.")
        st.markdown("**Motor**")
        st.caption("Sepeda motor, skuter, dan kendaraan roda dua lainnya")

    with col3:
        if os.path.exists("images/truck.jpg"):
            st.image("images/truck.jpg", use_container_width=True)
        else:
            st.warning("⚠️ Gambar truk tidak ditemukan. Pastikan file `images/truck.jpg` ada di folder `images`.")
        st.markdown("**Truk**")
        st.caption("Truk barang dan kendaraan niaga berat")

    st.markdown("---")
    st.info("Pilih menu **Prediksi** di sidebar untuk mencoba deteksi kendaraan.")

# ======================================
# Halaman PREDIKSI
# ======================================
elif selected == "Prediksi":
    st.header("🔍 Prediksi Jenis Kendaraan")
    uploaded_file = st.file_uploader("Unggah gambar kendaraan:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)

        st.write("🚧 *Model prediksi akan ditambahkan di sini nanti.*")
    else:
        st.write("Silakan unggah gambar kendaraan untuk memulai prediksi.")

# ======================================
# Halaman TENTANG
# ======================================
elif selected == "Tentang":
    st.header("Tentang Aplikasi")
    st.write("""
        Aplikasi ini dibuat untuk mendeteksi jenis kendaraan menggunakan model YOLO dan classifier CNN.
        Dikembangkan oleh tim **Vehicle-Type-Recognition 2025**.
        
        **Fitur Utama:**
        - Deteksi gambar kendaraan secara otomatis.
        - Tampilan interaktif menggunakan Streamlit.
        - Dapat dijalankan secara lokal maupun di Streamlit Cloud.
    """)

    st.success("Versi: 1.0.0 | Dibuat oleh: agnaaa")

