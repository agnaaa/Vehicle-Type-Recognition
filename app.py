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
# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="AI Image Detection",
    layout="wide"
)

# =============================
# CUSTOM CSS
# =============================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #fdeef2 0%, #fff 100%);
        font-family: 'Inter', sans-serif;
    }

    /* --- Navbar --- */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 4rem;
        background-color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border-radius: 0 0 20px 20px;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    .navbar-left {
        font-weight: 700;
        font-size: 20px;
        color: #1f2937;
    }
    .navbar-left span {
        color: #ec5c9a;
    }
    .navbar-right a {
        margin-left: 2rem;
        text-decoration: none;
        font-weight: 600;
        color: #1f2937;
        transition: 0.3s;
        padding: 6px 12px;
        border-radius: 8px;
    }
    .navbar-right a.active {
        background-color: #fde3ec;
        color: #ec5c9a;
    }
    .navbar-right a:hover {
        color: #ec5c9a;
    }

    /* --- Buttons --- */
    .btn-primary {
        background-color: #ec5c9a;
        border: none;
        color: white;
        font-weight: 600;
        padding: 0.8rem 1.8rem;
        border-radius: 10px;
        margin-top: 1.5rem;
        margin-right: 1rem;
        cursor: pointer;
    }
    .btn-outline {
        border: 2px solid #f4b7d0;
        background: none;
        color: #ec5c9a;
        font-weight: 600;
        padding: 0.8rem 1.8rem;
        border-radius: 10px;
        cursor: pointer;
    }

    /* --- Upload Card --- */
    .upload-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(236,92,154,0.15);
        width: 340px;
        text-align: center;
    }
    .upload-card h4 {
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 1rem;
    }

    /* --- Footer Button --- */
    .talk-btn {
        position: fixed;
        bottom: 25px;
        right: 25px;
        background-color: #ec5c9a;
        color: white;
        padding: 12px 24px;
        border-radius: 30px;
        font-weight: 600;
        text-decoration: none;
        box-shadow: 0 4px 15px rgba(236,92,154,0.4);
    }
    .talk-btn:hover {
        background-color: #e34c8f;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# NAVBAR (aktif berdasarkan halaman)
# =============================
if "page" not in st.session_state:
    st.session_state.page = "Home"

query_params = st.query_params
if "page" in query_params:
    st.session_state.page = query_params["page"].capitalize()

selected_page = st.session_state.page

nav_items = ["Home", "Classification"]
navbar_html = '<div class="navbar"><div class="navbar-left">AI <span>Image Detection</span></div><div class="navbar-right">'
for item in nav_items:
    active_class = "active" if item == selected_page else ""
    navbar_html += f'<a href="?page={item}" class="{active_class}">{item}</a>'
navbar_html += "</div></div>"
st.markdown(navbar_html, unsafe_allow_html=True)

# =============================
# HALAMAN HOME
# =============================
if selected_page == "Home":
    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;padding:5rem 6rem 3rem 6rem;">
        <div style="max-width:600px;">
            <h1 style="font-size:48px;font-weight:800;color:#1f2937;line-height:1.2;">Deteksi Jenis<br><span style="color:#ec5c9a;">Kendaraan AI</span></h1>
            <p style="font-size:16px;color:#6b7280;margin-top:1rem;line-height:1.6;">
                Platform revolusioner yang menggunakan teknologi deep learning
                untuk mengidentifikasi dan mengklasifikasi jenis kendaraan seperti mobil,
                motor, truck, dan bus dengan akurasi tinggi.
            </p>
            <button class="btn-primary">🚀 Coba Sekarang</button>
            <button class="btn-outline">📘 Pelajari Lebih Lanjut</button>
        </div>
        <div class="upload-card">
            <h4>Demo Cepat</h4>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload gambar kendaraan untuk analisis", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar kendaraan diunggah", use_container_width=True)
        st.success("✅ Gambar berhasil diunggah (Demo).")

    st.markdown("</div></div>", unsafe_allow_html=True)

# =============================
# HALAMAN CLASSIFICATION
# =============================
elif selected_page == "Classification":
    st.markdown("""
    <div style="text-align:center;margin-top:3rem;">
        <h2 style="font-size:36px;font-weight:800;color:#1f2937;">Klasifikasi Gambar AI</h2>
        <p style="color:#6b7280;font-size:16px;margin-bottom:2rem;">
            Upload gambar dan biarkan AI kami menganalisis serta mengklasifikasi objek dalam gambar dengan akurasi tinggi.
        </p>
    </div>

    <div style="display:flex;justify-content:center;gap:3rem;margin-top:2rem;">
        <div style="background:white;padding:2rem;border-radius:15px;box-shadow:0 10px 30px rgba(236,92,154,0.15);width:350px;text-align:center;">
            <h4>Upload Gambar</h4>
            <p style="color:#6b7280;">Pilih atau Drop Gambar<br><small>Mendukung JPG, PNG, WebP hingga 10MB</small></p>
        </div>

        <div style="background:white;padding:2rem;border-radius:15px;box-shadow:0 10px 30px rgba(236,92,154,0.15);width:350px;text-align:center;">
            <h4>Hasil Klasifikasi</h4>
            <p style="color:#6b7280;">Upload dan analisis gambar untuk melihat hasil klasifikasi</p>
        </div>
    </div>

    <div style="text-align:center;margin-top:3rem;">
        <h4>Coba Gambar Contoh</h4>
        <div style="display:flex;justify-content:center;gap:1.5rem;margin-top:1.5rem;">
            <div style="background:white;padding:1rem;border-radius:15px;width:150px;box-shadow:0 5px 15px rgba(236,92,154,0.15);">
                <img src="https://cdn-icons-png.flaticon.com/512/616/616408.png" width="80"><p>Kucing</p>
            </div>
            <div style="background:white;padding:1rem;border-radius:15px;width:150px;box-shadow:0 5px 15px rgba(236,92,154,0.15);">
                <img src="https://cdn-icons-png.flaticon.com/512/616/616408.png" width="80"><p>Anjing</p>
            </div>
            <div style="background:white;padding:1rem;border-radius:15px;width:150px;box-shadow:0 5px 15px rgba(236,92,154,0.15);">
                <img src="https://cdn-icons-png.flaticon.com/512/2331/2331717.png" width="80"><p>Mobil</p>
            </div>
            <div style="background:white;padding:1rem;border-radius:15px;width:150px;box-shadow:0 5px 15px rgba(236,92,154,0.15);">
                <img src="https://cdn-icons-png.flaticon.com/512/415/415682.png" width="80"><p>Buah</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================
# FOOTER BUTTON
# =============================
st.markdown("""
<a class="talk-btn" href="#">💬 Talk with Us</a>
""", unsafe_allow_html=True)
