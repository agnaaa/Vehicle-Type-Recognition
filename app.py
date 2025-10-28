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
# --- KONFIGURASI DASAR ---
st.set_page_config(
    page_title="Vehicle Type Recognition",
    page_icon="🚗",
    layout="wide"
)

# --- NAVBAR SEDERHANA ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Home", "Tentang Pengembang"])

# ===============================================================
# ======================== HOME PAGE =============================
# ===============================================================
if page == "Home":
    st.markdown(
        """
        <h1 style='text-align:center; color:#2C3E50;'>🚘 Vehicle Type Recognition</h1>
        <p style='text-align:center; font-size:18px; color:#555;'>
        Sistem deteksi jenis kendaraan berbasis AI dengan akurasi tinggi, efisiensi waktu, 
        dan kemudahan penggunaan untuk berbagai kebutuhan analisis visual.
        </p>
        """, unsafe_allow_html=True
    )

    st.markdown("---")

    # --- Bagian Alasan Memilih Platform ---
    st.markdown(
        """
        <h2 style='color:#E91E63;'>Mengapa Memilih Platform Ini?</h2>
        <ul style='font-size:17px; color:#444;'>
            <li><b>Akurasi Tinggi:</b> Model deep learning yang dilatih dengan ribuan data kendaraan.</li>
            <li><b>Efisiensi Waktu:</b> Deteksi kendaraan hanya dalam hitungan detik.</li>
            <li><b>Kemudahan Penggunaan:</b> Antarmuka sederhana tanpa perlu keahlian teknis.</li>
            <li><b>Dukungan Multi-Kendaraan:</b> Dapat mengenali mobil, motor, bus, truk, dan lainnya.</li>
        </ul>
        """, unsafe_allow_html=True
    )

    st.markdown("---")

    # --- Jenis Kendaraan yang Bisa Dideteksi ---
    st.markdown(
        """
        <h2 style='color:#E91E63;'>Jenis Kendaraan yang Dapat Dideteksi</h2>
        <div style='display:flex; justify-content:space-around; margin-top:20px;'>
            <div style='text-align:center;'>
                🚗 <br>Mobil
            </div>
            <div style='text-align:center;'>
                🏍️ <br>Motor
            </div>
            <div style='text-align:center;'>
                🚚 <br>Truk
            </div>
            <div style='text-align:center;'>
                🚌 <br>Bus
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("---")

    # --- Bagian Kolaborasi ---
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #f59ac4, #e66aa5);
            padding: 50px;
            border-radius: 20px;
            text-align: center;
            color: white;
            margin-top: 40px;
        ">
            <h2><b>Tertarik Berkolaborasi?</b></h2>
            <p style="font-size:17px; margin-bottom: 30px;">
                Kami selalu terbuka untuk kolaborasi penelitian, partnership, 
                atau diskusi tentang implementasi teknologi AI dalam proyek Anda. 
                Mari bersama-sama menciptakan masa depan yang lebih cerdas!
            </p>
            <a href="https://wa.me/6289669727601" target="_blank">
                <button style="
                    background-color:white; 
                    color:#e91e63; 
                    border:none; 
                    padding:12px 25px; 
                    border-radius:10px; 
                    cursor:pointer; 
                    font-size:16px;
                ">
                    💬 Hubungi Tim Research
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True
    )


# ===============================================================
# ===================== ABOUT PAGE ===============================
# ===============================================================
elif page == "Tentang Pengembang":
    st.markdown(
        "<h1 style='text-align:center; color:#2C3E50;'>👩‍💻 Pengembang</h1>",
        unsafe_allow_html=True
    )

    # --- Tampilan Dua Kolom ---
    col1, col2 = st.columns([1, 2])

    # Foto Agna
    image_path = "6372789C-781F-4439-AE66-2187B96D6952.jpeg"
    try:
        with col1:
            image = Image.open(image_path)
            st.image(image, caption="Agna Balqis", width=280)
    except FileNotFoundError:
        with col1:
            st.warning("⚠️ Foto Agna tidak ditemukan. Pastikan file ada di direktori utama.")

    # Deskripsi Agna
    with col2:
        st.markdown(
            """
            ### Agna Balqis  
            **Lead AI Developer**  

            Membangun sistem AI dan antarmuka pengguna dengan fokus pada akurasi, efisiensi, 
            serta kemudahan penggunaan.  
            Menggabungkan teknologi modern untuk menciptakan pengalaman deteksi kendaraan 
            yang cerdas dan interaktif.
            """, unsafe_allow_html=True
        )

        # Tombol WA langsung ke nomor Agna
        wa_url = "https://wa.me/6289669727601"
        st.markdown(
            f"""
            <a href="{wa_url}" target="_blank">
                <button style="
                    background-color:#e91e63;
                    color:white;
                    border:none;
                    padding:10px 22px;
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
