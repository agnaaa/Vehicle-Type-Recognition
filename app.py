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
# ==============================
# Tampilan Halaman Utama
# ==============================
def show_home():
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(180deg, #fff5f8, #ffffff);
            font-family: 'Inter', sans-serif;
        }
        h1 span {
            color: #f07da7;
        }
        .subtitle {
            color: #6b7280;
            font-size: 16px;
            line-height: 1.6;
        }
        .btn-primary {
            background: linear-gradient(90deg, #f07da7, #e86e9a);
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            font-weight: 600;
            border: none;
        }
        .btn-outline {
            border: 1px solid #f6cde0;
            color: #f07da7;
            padding: 10px 20px;
            border-radius: 10px;
            font-weight: 600;
        }
        .card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(16,24,40,0.06);
            text-align: center;
            padding: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Deteksi Jenis 🚗 Kendaraan **AI**")
    st.markdown(
        "<p class='subtitle'>Platform revolusioner yang menggunakan teknologi deep learning "
        "untuk mengidentifikasi dan mengklasifikasikan jenis kendaraan seperti mobil, motor, truck, "
        "dan bus dengan akurasi tinggi.</p>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<button class='btn-primary'>🚀 Coba Sekarang</button>", unsafe_allow_html=True)
        st.markdown("<button class='btn-outline'>📘 Pelajari Lebih Lanjut</button>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'><h4>Demo Cepat</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload gambar kendaraan untuk analisis", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar kendaraan diunggah", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("---")
    st.subheader("Jenis Kendaraan yang Dapat Dideteksi")
    st.markdown("<p style='color:#6b7280'>Sistem AI kami dapat mengenali berbagai jenis kendaraan dengan akurasi tinggi</p>", unsafe_allow_html=True)

    colA, colB, colC, colD = st.columns(4)
    colA.image("https://via.placeholder.com/200x100?text=Mobil")
    colA.caption("Mobil – Sedan, SUV, Hatchback, dan berbagai jenis mobil penumpang")

    colB.image("https://via.placeholder.com/200x100?text=Motor")
    colB.caption("Motor – Sepeda motor, skuter, dan kendaraan roda dua lainnya")

    colC.image("https://via.placeholder.com/200x100?text=Truck")
    colC.caption("Truck – Truk kargo, pickup, dan kendaraan komersial berat")

    colD.image("https://via.placeholder.com/200x100?text=Bus")
    colD.caption("Bus – Bus kota, antar kota, dan kendaraan angkutan umum")

    st.write("---")
    st.subheader("Mengapa Memilih Platform Kami?")
    st.markdown("<p style='color:#6b7280'>Teknologi AI terdepan yang dirancang khusus untuk deteksi kendaraan dengan akurasi tinggi.</p>", unsafe_allow_html=True)

    colx, coly, colz, cola = st.columns(4)
    colx.metric("Akurasi Model", "98.2%")
    coly.metric("Waktu Proses", "47ms")
    colz.metric("Jenis Kendaraan", "4+")
    cola.metric("Uptime", "99.9%")

# Jalankan halaman
if __name__ == "__main__":
    show_home()
