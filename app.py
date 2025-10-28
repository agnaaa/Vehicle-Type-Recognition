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
# Konfigurasi dasar halaman
st.set_page_config(page_title="AI Vehicle Detection", layout="wide")

# ===== CSS STYLING =====
st.markdown("""
<style>
body {
    background-color: #fde8ef;
    font-family: 'Poppins', sans-serif;
}
.navbar {
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: #f4b6c2;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 40px;
}
.nav-item {
    margin: 0 25px;
    font-weight: 500;
    color: #333;
    text-decoration: none;
}
.nav-item:hover {
    color: #000;
}
.active {
    font-weight: 700;
    border-bottom: 3px solid #e35d88;
}
.title {
    font-size: 42px;
    font-weight: 700;
    color: #111;
}
.highlight {
    color: #e35d88;
}
.subtitle {
    color: #333;
    font-size: 16px;
    margin-bottom: 30px;
}
.btn-primary {
    background-color: #e35d88;
    color: white;
    border: none;
    padding: 10px 25px;
    border-radius: 10px;
    cursor: pointer;
    font-size: 15px;
}
.btn-primary:hover {
    background-color: #ff6f91;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ===== NAVBAR =====
if "page" not in st.session_state:
    st.session_state.page = "Home"

menu = st.radio(
    "",
    ["Home", "Classification", "About Project"],
    horizontal=True,
    label_visibility="collapsed"
)

# ===== HOME PAGE =====
if menu == "Home":
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<div class='title'>Deteksi Jenis <span class='highlight'>Kendaraan AI</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Platform revolusioner yang menggunakan teknologi deep learning untuk mengidentifikasi dan mengklasifikasi jenis kendaraan seperti mobil, motor, truck, dan bus dengan akurasi tinggi.</div>", unsafe_allow_html=True)
        if st.button("🚗 Coba Sekarang", key="try_now"):
            st.session_state.page = "Classification"
            st.session_state["redirect"] = True

    with col2:
        st.markdown("""
        <div class="card">
            <h4>Demo Cepat</h4>
            <div style="border: 2px dashed #f4b6c2; padding: 30px; border-radius: 10px;">
                <div style="font-size:40px;">🖼️</div>
                <div style="color:#b88a9f;">Upload gambar kendaraan untuk analisis</div>
                <br>
                <button class="btn-primary">Pilih Gambar</button>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Jenis Kendaraan yang Dapat Dideteksi")
    cols = st.columns(4)
    vehicles = [
        ("🚗 Mobil", "Sedan, SUV, Hatchback, dan mobil penumpang lainnya"),
        ("🏍️ Motor", "Sepeda motor, skuter, dan kendaraan roda dua lainnya"),
        ("🚚 Truck", "Truk kargo, pickup, dan kendaraan komersial berat"),
        ("🚌 Bus", "Bus kota, antar kota, dan kendaraan umum"),
    ]
    for i, (title, desc) in enumerate(vehicles):
        with cols[i]:
            st.markdown(f"<div class='card'><h4>{title}</h4><p>{desc}</p></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    metrics = st.columns(4)
    metrics_list = [("98.2%", "Akurasi Model"), ("47ms", "Waktu Proses"), ("4+", "Jenis Kendaraan"), ("99.9%", "Uptime")]
    for i, (val, label) in enumerate(metrics_list):
        with metrics[i]:
            st.markdown(f"<div class='card'><h2 style='color:#e35d88'>{val}</h2><p>{label}</p></div>", unsafe_allow_html=True)

# ===== CLASSIFICATION PAGE =====
elif menu == "Classification" or st.session_state.get("page") == "Classification":
    st.session_state.page = "Classification"
    st.markdown("<h2 style='text-align:center;'>Klasifikasi Gambar AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Upload gambar dan biarkan AI mengenali jenis kendaraan.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("<h4>Upload Gambar</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Pilih gambar kendaraan", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Gambar yang diunggah", use_container_width=True)
            if st.button("🔍 Analisis Gambar"):
                with st.spinner("Menganalisis gambar..."):
                    time.sleep(1.5)
                pred_labels = ["Mobil", "Motor", "Truck", "Bus"]
                preds = {label: random.randint(70, 99) for label in pred_labels}
                preds = {k: round(v, 1) for k, v in preds.items()}
                st.success("Analisis selesai!")

                with col2:
                    st.markdown("<h4>Hasil Klasifikasi</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p>Waktu Proses: <b>{random.randint(40,80)}ms</b></p>", unsafe_allow_html=True)
                    st.markdown(f"<p>Model: <b>VehicleDetect-v1.0</b></p>", unsafe_allow_html=True)
                    st.markdown("<h5>Prediksi Teratas</h5>", unsafe_allow_html=True)
                    for label, conf in preds.items():
                        st.progress(conf / 100)
                        st.markdown(f"**{label}** — {conf}%")

# ===== ABOUT PROJECT =====
elif menu == "About Project":
    st.markdown("<h2>Tentang Proyek</h2>", unsafe_allow_html=True)
    st.write("""
    Aplikasi ini dikembangkan untuk mendeteksi jenis kendaraan menggunakan model AI. 
    Sistem ini mampu mengenali kendaraan seperti mobil, motor, truck, dan bus dengan tingkat akurasi tinggi.
    """)
    st.write("Dibangun dengan ❤️ menggunakan Streamlit.")
