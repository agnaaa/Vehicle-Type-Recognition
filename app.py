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
# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="AI Image Detection", layout="wide")

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
    <style>
        body {
            background-color: #fdeef4;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #fdeef4;
        }
        [data-testid="stHeader"] {
            background-color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        /* Navbar */
        .navbar {
            display: flex;
            justify-content: center;
            gap: 40px;
            padding: 1rem 0;
            background-color: white;
            border-bottom: 1px solid #f5d0dc;
        }
        .nav-item {
            font-weight: 500;
            color: #333;
            text-decoration: none;
            padding: 6px 12px;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .nav-item.active, .nav-item:hover {
            background-color: #f9c6d3;
            color: #000;
        }

        /* Tombol pink gradient */
        .btn-pink {
            background: linear-gradient(135deg, #f06292, #ec407a);
            color: white !important;
            border: none;
            padding: 10px 22px;
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
            text-decoration: none;
        }
        .btn-pink:hover {
            background: linear-gradient(135deg, #ec407a, #f06292);
            transform: translateY(-2px);
        }
        .btn-outline {
            border: 2px solid #f06292;
            color: #f06292;
            padding: 10px 22px;
            border-radius: 10px;
            font-weight: 600;
            text-decoration: none;
        }
        .btn-outline:hover {
            background-color: #f0629210;
        }

        /* Kartu Upload */
        .upload-card {
            background-color: white;
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            width: 400px;
        }

        /* Floating Chat */
        .chat-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #c2185b;
            color: white;
            padding: 12px 22px;
            border-radius: 50px;
            font-weight: 600;
            text-decoration: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .chat-btn:hover {
            background: #ad1457;
        }

        /* Animasi transisi */
        .fade-in {
            animation: fadeIn 0.6s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- NAVIGATION ----------------------
st.session_state.setdefault("page", "Home")

col1, col2, col3 = st.columns([3,4,3])
with col2:
    st.markdown("""
        <div class="navbar">
            <a href="#" class="nav-item {home_active}" onclick="window.location.href='?page=home'">Home</a>
            <a href="#" class="nav-item {class_active}" onclick="window.location.href='?page=classification'">Classification</a>
            <a href="#" class="nav-item" onclick="window.location.href='?page=about'">About Project</a>
        </div>
    """.format(
        home_active="active" if st.session_state.page == "Home" else "",
        class_active="active" if st.session_state.page == "Classification" else ""
    ), unsafe_allow_html=True)

# ---------------------- PAGE CONTROL ----------------------
query_params = st.experimental_get_query_params()
if "page" in query_params:
    st.session_state.page = query_params["page"][0].capitalize()

# ---------------------- HOME PAGE ----------------------
if st.session_state.page == "Home":
    st.markdown("""
    <div class="fade-in" style="padding: 40px 0;">
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap;">
            <div style="flex:1; min-width:300px;">
                <h1 style="font-size:45px; color:#212121; font-weight:800;">Deteksi Jenis<br><span style="color:#ec407a;">Kendaraan AI</span></h1>
                <p style="font-size:17px; color:#444; max-width:500px;">
                    Platform revolusioner yang menggunakan teknologi deep learning untuk mengidentifikasi dan mengklasifikasi jenis kendaraan seperti mobil, motor, truck, dan bus dengan akurasi tinggi.
                </p>
                <div style="margin-top:25px;">
                    <a class="btn-pink" href="?page=classification">🚀 Coba Sekarang</a>
                    <a class="btn-outline" href="#learn-more">📘 Pelajari Lebih Lanjut</a>
                </div>
            </div>
            <div class="upload-card">
                <h4><b>Demo Cepat</b></h4>
                <p>Upload gambar kendaraan untuk analisis</p>
                <input type="file" style="margin-top:10px;">
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------- CLASSIFICATION PAGE ----------------------
elif st.session_state.page == "Classification":
    st.markdown("""
    <div class="fade-in" style="text-align:center; margin-top:40px;">
        <h2 style="font-weight:800;">Klasifikasi Gambar AI</h2>
        <p>Upload gambar dan biarkan AI kami menganalisis serta mengklasifikasi objek dalam gambar dengan akurasi tinggi</p>
    </div>
    <div style="display:flex; justify-content:center; gap:50px; margin-top:40px; flex-wrap:wrap;">
        <div class="upload-card">
            <h4><b>Upload Gambar</b></h4>
            <p>Pilih atau Drop Gambar<br><small>Mendukung JPG, PNG, WebP hingga 10MB</small></p>
            <input type="file" style="margin-top:10px;">
        </div>
        <div class="upload-card">
            <h4><b>Hasil Klasifikasi</b></h4>
            <p>Upload dan analisis gambar untuk melihat hasil klasifikasi</p>
        </div>
    </div>
    <div style="text-align:center; margin-top:60px;">
        <h4>Coba Gambar Contoh</h4>
        <div style="display:flex; justify-content:center; gap:30px; margin-top:20px; flex-wrap:wrap;">
            <div><img src="https://i.imgur.com/1kX5aQF.png" width="100"><p>Kucing</p></div>
            <div><img src="https://i.imgur.com/EnpY0Hs.png" width="100"><p>Anjing</p></div>
            <div><img src="https://i.imgur.com/Zq5eZBh.png" width="100"><p>Mobil</p></div>
            <div><img src="https://i.imgur.com/pKNKPBp.png" width="100"><p>Buah</p></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------- FLOATING CHAT BUTTON ----------------------
st.markdown('<a href="#" class="chat-btn">💬 Talk with Us</a>', unsafe_allow_html=True)
