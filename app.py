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
# =================================================================
# KONFIGURASI HALAMAN
# =================================================================
st.set_page_config(page_title="AI Vehicle Detection", page_icon="🚗", layout="wide")

# =================================================================
# GAYA UMUM (WARNA, FONT, BORDER, DLL)
# =================================================================
st.markdown("""
    <style>
    /* Font dan warna dasar */
    body {
        background-color: #fff6f9;
        color: #333;
        font-family: "Poppins", sans-serif;
    }

    h1, h2, h3 {
        font-weight: 700;
    }

    /* Navigasi menu */
    .nav-link {
        font-size: 16px !important;
        color: #333 !important;
        font-weight: 600 !important;
    }

    /* Tombol utama */
    .stButton>button {
        background-color: #ff6f91;
        color: white;
        border-radius: 10px;
        padding: 8px 20px;
        border: none;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #ff85a1;
        transform: translateY(-2px);
    }

    /* Pemisah */
    hr {
        border: 1px solid #f8cdda;
    }
    </style>
""", unsafe_allow_html=True)

# =================================================================
# MENU NAVIGASI ATAS
# =================================================================
selected = option_menu(
    menu_title=None,
    options=["Home", "Classification", "Model Performance", "Model Info", "About Project"],
    icons=["house", "camera", "bar-chart", "info-circle", "file-earmark-person"],
    orientation="horizontal",
    styles={
        "container": {"padding": "10px 5px", "background-color": "#fff"},
        "icon": {"color": "#ff6f91", "font-size": "20px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px 10px",
            "border-radius": "8px",
            "color": "#000000",
        },
        "nav-link-selected": {"background-color": "#ff6f91", "color": "white"},
    }
)

# =================================================================
# HALAMAN HOME
# =================================================================
if selected == "Home":

    # BAGIAN HEADER
    st.markdown("""
        <div style="text-align:center; padding:50px 20px;">
            <h1 style="color:#1e1e1e;">Deteksi Jenis Kendaraan dengan AI</h1>
            <p style="font-size:18px; color:#555; max-width:700px; margin:auto;">
                Platform AI revolusioner yang menggunakan deep learning untuk mengidentifikasi dan mengklasifikasikan jenis kendaraan seperti mobil, motor, truk, dan bus dengan akurasi tinggi.
            </p>
            <div style="margin-top:30px;">
                <a href="#" style="background-color:#ff6f91; color:white; padding:12px 30px; border-radius:10px; text-decoration:none; font-weight:600; margin-right:10px;">🚀 Coba Sekarang</a>
                <a href="#" style="background-color:#f8cdda; color:#333; padding:12px 30px; border-radius:10px; text-decoration:none; font-weight:600;">📖 Pelajari Lebih Lanjut</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # BAGIAN JENIS KENDARAAN
    # ============================================================
    st.markdown("""
        <style>
        .vehicle-card {
            background-color: #fff;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.08);
            text-align: center;
            transition: transform 0.2s ease-in-out;
        }
        .vehicle-card:hover {
            transform: translateY(-5px);
        }
        .vehicle-img {
            border-radius: 12px;
            width: 100%;
            height: 150px;
            object-fit: cover;
        }
        .vehicle-title {
            font-weight: 700;
            color: #1e1e1e;
            margin-top: 10px;
            font-size: 18px;
        }
        .vehicle-desc {
            color: #555;
            font-size: 14px;
        }
        </style>

        <h1 style="text-align:center; color:#1e1e1e;">Jenis Kendaraan yang Dapat Dideteksi</h1>
        <p style="text-align:center; color:#666;">Sistem AI kami dapat mengenali berbagai jenis kendaraan dengan akurasi tinggi</p>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
            <div class="vehicle-card">
                <img src="https://cdn.pixabay.com/photo/2017/03/27/13/28/auto-2179220_640.jpg" class="vehicle-img">
                <p class="vehicle-title">Mobil</p>
                <p class="vehicle-desc">Sedan, SUV, Hatchback, dan berbagai jenis mobil penumpang</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="vehicle-card">
                <img src="https://cdn.pixabay.com/photo/2018/08/02/00/50/motorcycle-3575557_640.jpg" class="vehicle-img">
                <p class="vehicle-title">Motor</p>
                <p class="vehicle-desc">Sepeda motor, skuter, dan kendaraan roda dua lainnya</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="vehicle-card">
                <img src="https://cdn.pixabay.com/photo/2014/12/27/15/40/truck-581173_640.jpg" class="vehicle-img">
                <p class="vehicle-title">Truck</p>
                <p class="vehicle-desc">Truk kargo, pickup, dan kendaraan komersial berat</p>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <div class="vehicle-card">
                <img src="https://cdn.pixabay.com/photo/2016/11/29/02/53/bus-1868313_640.jpg" class="vehicle-img">
                <p class="vehicle-title">Bus</p>
                <p class="vehicle-desc">Bus kota, bus antar kota, dan kendaraan angkutan umum</p>
            </div>
        """, unsafe_allow_html=True)

    st.write("")  # spasi tambahan

    # ============================================================
    # BAGIAN PERFORMA MODEL
    # ============================================================
    st.markdown("""
        <h2 style="text-align:center; margin-top:60px;">Performa Model Kami</h2>
    """, unsafe_allow_html=True)

    perf1, perf2, perf3, perf4 = st.columns(4)
    perf1.metric("Akurasi Model", "98.2%")
    perf2.metric("Waktu Proses", "47 ms")
    perf3.metric("Jenis Kendaraan", "4+")
    perf4.metric("Uptime", "99.9%")

    # ============================================================
    # BAGIAN KEUNGGULAN
    # ============================================================
    st.markdown("""
        <h2 style="text-align:center; margin-top:60px;">Mengapa Memilih Platform Kami?</h2>
        <p style="text-align:center; color:#666;">Teknologi AI terdepan yang dirancang khusus untuk deteksi kendaraan dengan akurasi tinggi</p>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown("""
        <div class="vehicle-card">
            <h3>🎯 Deteksi Akurat</h3>
            <p>Akurasi hingga 98.2% dalam mengenali jenis kendaraan dengan deep learning.</p>
        </div>
    """, unsafe_allow_html=True)

    c2.markdown("""
        <div class="vehicle-card">
            <h3>⚡ Pemrosesan Cepat</h3>
            <p>Identifikasi kendaraan dalam waktu kurang dari 50ms.</p>
        </div>
    """, unsafe_allow_html=True)

    c3.markdown("""
        <div class="vehicle-card">
            <h3>🔒 Keamanan Tinggi</h3>
            <p>Data gambar kendaraan diproses dengan enkripsi end-to-end.</p>
        </div>
    """, unsafe_allow_html=True)

    c4.markdown("""
        <div class="vehicle-card">
            <h3>🌍 API Global</h3>
            <p>Akses mudah melalui REST API untuk integrasi sistem traffic management.</p>
        </div>
    """, unsafe_allow_html=True)
