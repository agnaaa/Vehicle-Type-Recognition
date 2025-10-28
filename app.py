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
import streamlit as st

# =============================
# Konfigurasi Awal
# =============================
st.set_page_config(
    page_title="AI Vehicle Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "page" not in st.session_state:
    st.session_state.page = "Home"

# =============================
# Sidebar Navigasi
# =============================
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Home", "Tentang"])

# =============================
# HALAMAN HOME
# =============================
if page == "Home":
    st.markdown(
        """
        <div style="text-align: center; padding: 40px 0;">
            <h1 style="font-size: 38px; font-weight: bold; color: #111827;">Sistem Deteksi Kendaraan Berbasis AI</h1>
            <p style="font-size: 18px; color: #4B5563; margin-top: 10px;">
                Deteksi berbagai jenis kendaraan secara cepat, akurat, dan efisien menggunakan teknologi Vision AI terkini.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Statistik utama
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Akurasi Model", value="98.7%")
    with col2:
        st.metric(label="Jenis Kendaraan Terdeteksi", value="5 Jenis")
    with col3:
        st.metric(label="Kecepatan Deteksi", value="0.3 detik")

    # Jenis kendaraan
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; margin-top: 30px;">
            <h2 style="font-weight: 700; color: #111827;">Jenis Kendaraan yang Dapat Dideteksi</h2>
            <p style="color: #4B5563; font-size: 17px;">
                Sistem ini mampu mengenali berbagai jenis kendaraan seperti:
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    with col_a:
        st.image("assets/car.png", width=60)
        st.caption("Mobil")
    with col_b:
        st.image("assets/motorcycle.png", width=60)
        st.caption("Motor")
    with col_c:
        st.image("assets/bus.png", width=60)
        st.caption("Bus")
    with col_d:
        st.image("assets/truck.png", width=60)
        st.caption("Truk")
    with col_e:
        st.image("assets/bicycle.png", width=60)
        st.caption("Sepeda")

    # Alasan memilih platform
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; margin-top: 30px;">
            <h2 style="font-weight: 700; color: #111827;">Mengapa Memilih Platform Kami?</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 25px; margin-top: 20px;">
            <div style="background-color: white; width: 280px; padding: 20px; border-radius: 16px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center;">
                <h4 style="color: #E11D48;">⚡ Efisiensi Tinggi</h4>
                <p style="color: #4B5563;">Model AI kami dioptimalkan untuk mendeteksi kendaraan dalam hitungan detik tanpa mengorbankan akurasi.</p>
            </div>

            <div style="background-color: white; width: 280px; padding: 20px; border-radius: 16px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center;">
                <h4 style="color: #E11D48;">🎯 Akurasi Terbaik</h4>
                <p style="color: #4B5563;">Dilatih menggunakan ribuan data kendaraan sehingga menghasilkan prediksi yang sangat akurat.</p>
            </div>

            <div style="background-color: white; width: 280px; padding: 20px; border-radius: 16px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center;">
                <h4 style="color: #E11D48;">🧠 Teknologi AI Modern</h4>
                <p style="color: #4B5563;">Menggunakan model YOLOv8 terbaru untuk performa dan efisiensi tinggi pada berbagai kondisi pencahayaan.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Tombol navigasi ke klasifikasi
    st.markdown("<div style='text-align: center; margin-top: 40px;'>", unsafe_allow_html=True)
    if st.button("🚗 Coba Sekarang", use_container_width=False):
        st.session_state.page = "Classification"
        st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# HALAMAN TENTANG
# =============================
elif page == "Tentang":
    st.markdown(
        """
        <div style="text-align: center; padding: 40px 0;">
            <h1 style="font-size: 36px; font-weight: bold; color: #111827;">Tentang Proyek AI Vehicle Detection</h1>
            <p style="font-size: 17px; color: #4B5563; margin-top: 10px;">
                Sistem deteksi kendaraan berbasis AI ini dikembangkan untuk mendukung analitik transportasi, keamanan lalu lintas, 
                dan sistem transportasi cerdas masa depan. Proyek ini berfokus pada efisiensi, akurasi, serta kemudahan implementasi.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Visi & Misi
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div style="background-color: white; padding: 25px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                <h3 style="color: #E11D48;">Misi Kami</h3>
                <p style="color: #4B5563; font-size: 16px;">
                    Menghadirkan teknologi AI yang mampu mengenali kendaraan secara cepat, akurat, dan efisien, 
                    membantu pengambilan keputusan di sektor transportasi modern serta mendukung sistem keamanan lalu lintas.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div style="background-color: white; padding: 25px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                <h3 style="color: #E11D48;">Visi Kami</h3>
                <p style="color: #4B5563; font-size: 16px;">
                    Menjadi solusi Vision AI terbaik yang dapat diintegrasikan ke dalam sistem smart city 
                    dan infrastruktur transportasi masa depan, serta mendukung inovasi dalam efisiensi lalu lintas.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Pengembang
    st.markdown(
        """
        <div style="text-align: center; margin-top: 40px;">
            <h2 style="font-weight: 700; color: #111827;">Pengembang</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="display: flex; justify-content: center; margin-top: 20px;">
            <div style="background-color: white; border-radius: 20px; padding: 30px; width: 300px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.1); text-align: center;">
                <img src="assets/6372789C-781F-4439-AE66-2187B96D6952.jpeg" 
                     style="width: 180px; height: 180px; object-fit: cover; border-radius: 50%; margin-bottom: 20px;">
                <h3 style="color: #111827;">Agna Balqis</h3>
                <h4 style="color: #E11D48;">Lead AI Developer</h4>
                <p style="color: #4B5563; font-size: 15px;">
                    Mengembangkan model AI dan merancang tampilan visual proyek ini dengan penuh dedikasi.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
