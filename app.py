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
# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Vehicle Type Recognition",
    page_icon="🚗",
    layout="wide"
)

# --- Navbar Sederhana ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Home", "Tentang"])

# --- HOME PAGE ---
if page == "Home":
    st.title("🚘 Vehicle Type Recognition")
    st.write(
        """
        Aplikasi ini dirancang untuk mengenali jenis kendaraan secara otomatis 
        menggunakan model kecerdasan buatan (AI). 
        Unggah gambar kendaraan Anda dan biarkan sistem mendeteksi tipenya.
        """
    )

# --- ABOUT PAGE ---
elif page == "Tentang":
    st.title("👩‍💻 Tentang Pengembang")

    # Load foto Agna (pastikan nama file sama persis)
    image_path = "6372789C-781F-4439-AE66-2187B96D6952.jpeg"
    try:
        image = Image.open(image_path)
        st.image(image, caption="Agna Balqis", width=280)
    except FileNotFoundError:
        st.warning("⚠️ Foto Agna tidak ditemukan. Pastikan file ada di direktori utama.")

    st.markdown(
        """
        ### Agna Balqis  
        **Lead AI Developer**  

        Membangun sistem AI dan antarmuka pengguna dengan fokus pada akurasi, efisiensi, 
        serta kemudahan penggunaan.  
        Menggabungkan teknologi modern untuk menciptakan pengalaman deteksi kendaraan 
        yang cerdas dan interaktif.
        """
    )

    # Tombol langsung ke WhatsApp
    wa_url = "https://wa.me/6289669727601"
    st.markdown(
        f"""
        <a href="{wa_url}" target="_blank">
            <button style="
                background-color:#ff4b8b;
                color:white;
                border:none;
                padding:10px 20px;
                border-radius:10px;
                cursor:pointer;
                font-size:16px;
            ">
                💬 Hubungi Agna
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )
