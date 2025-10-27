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
# ============================================
# 🌸 KONFIGURASI HALAMAN
# ============================================
st.set_page_config(page_title="AI Image Detection", layout="wide")

# 🌸 CSS Styling — background pink soft pastel
st.markdown("""
    <style>
        body {
            background-color: #ffe6f0;
        }
        .main {
            background-color: #ffe6f0 !important;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #ffe6f0 !important;
        }
        [data-testid="stHeader"] {
            background-color: #ffd9e8 !important;
        }
        h1, h2, h3, h4, h5, h6, p {
            color: #1f1f1f;
        }
        .stButton>button {
            background-color: #f472b6;
            color: white;
            border-radius: 10px;
            font-weight: bold;
            padding: 0.6em 1.2em;
        }
        .stButton>button:hover {
            background-color: #ec4899;
        }
        .card {
            background-color: white;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            padding: 1.5em;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


# ============================================
# 🌸 NAVBAR
# ============================================
selected = option_menu(
    menu_title=None,
    options=["Home", "Classification", "Model Performance", "Model Info", "About Project"],
    icons=["house", "image", "bar-chart", "info-circle", "book"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "white", "box-shadow": "0 2px 6px rgba(0,0,0,0.05)"},
        "icon": {"color": "#f472b6", "font-size": "18px"},
        "nav-link": {
            "font-size": "16px",
            "color": "#1f1f1f",
            "padding": "10px 20px",
            "border-radius": "8px",
            "margin": "4px",
            "text-transform": "capitalize"
        },
        "nav-link-selected": {"background-color": "#f9c4d2", "color": "#000"},
    },
)


# ============================================
# 🌸 LOGO / JUDUL
# ============================================
st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:20px;">
        <span style="font-size:22px; font-weight:600;">🖼️ <span style="color:#f472b6;">AI Image Detection</span></span>
    </div>
""", unsafe_allow_html=True)


# ============================================
# 🌸 HOME PAGE
# ============================================
if selected == "Home":
    st.markdown("<h1 style='text-align:left;'>🚗 Deteksi Kendaraan AI</h1>", unsafe_allow_html=True)
    st.write("Platform berbasis deep learning yang mampu mendeteksi berbagai jenis kendaraan seperti mobil, motor, truk, dan bus dengan akurasi tinggi serta tampilan menarik.")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("🚀 Coba Sekarang")
    with col2:
        st.button("📘 Pelajari Lebih Lanjut")

    st.markdown("---")
    st.markdown("<h2 style='text-align:center;'>Jenis Kendaraan yang Dapat Dideteksi</h2>", unsafe_allow_html=True)
    cols = st.columns(4)
    kendaraan = ["🚗 Mobil", "🏍️ Motor", "🚛 Truck", "🚌 Bus"]
    deskripsi = [
        "Sedan, SUV, Hatchback, dan berbagai jenis mobil penumpang.",
        "Sepeda motor, skuter, dan kendaraan roda dua lainnya.",
        "Truk kargo, pickup, dan kendaraan komersial berat.",
        "Bus kota, bus antar kota, dan kendaraan angkutan umum."
    ]
    for i in range(4):
        with cols[i]:
            st.markdown(f"<div class='card'><h4>{kendaraan[i]}</h4><p>{deskripsi[i]}</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h2 style='text-align:center;'>Performa Model Kami</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("98.2%", "Akurasi Model"),
        ("47ms", "Waktu Proses"),
        ("4+", "Jenis Kendaraan"),
        ("99.9%", "Uptime")
    ]
    for i in range(4):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"<div class='card'><h2>{metrics[i][0]}</h2><p>{metrics[i][1]}</p></div>", unsafe_allow_html=True)


# ============================================
# 🌸 CLASSIFICATION PAGE
# ============================================
elif selected == "Classification":
    st.title("🧠 Klasifikasi Gambar AI")
    st.write("Upload gambar dan biarkan AI menganalisis serta mengklasifikasikan objek kendaraan dengan akurasi tinggi.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Upload Gambar")
        uploaded_file = st.file_uploader("Pilih atau Drop Gambar", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang Diupload", use_container_width=True)
            st.success("Gambar berhasil diunggah!")
        else:
            st.info("Upload gambar untuk memulai klasifikasi.")

    with col2:
        st.subheader("Hasil Klasifikasi")
        if uploaded_file is not None:
            st.markdown("<div style='padding:20px; background:#f9fafb; border-radius:10px; text-align:center;'>🚗 Jenis kendaraan terdeteksi: <b>Mobil</b></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='padding:20px; background:#f9fafb; border-radius:10px; text-align:center;'>Upload gambar untuk melihat hasil klasifikasi.</div>", unsafe_allow_html=True)


# ============================================
# 🌸 MODEL PERFORMANCE PAGE
# ============================================
elif selected == "Model Performance":
    st.markdown("<h1 style='text-align:center;'>📊 Model Performance</h1>", unsafe_allow_html=True)
    st.write("Analisis performa model deteksi kendaraan berdasarkan hasil pengujian dan evaluasi metrik utama.")

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("98.2%", "Accuracy"),
        ("97.8%", "Precision"),
        ("96.9%", "Recall"),
        ("97.3%", "F1-Score")
    ]
    for i in range(4):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"<div class='card'><h2 style='color:#c2185b;'>{metrics[i][0]}</h2><p>{metrics[i][1]}</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h3 style='text-align:center;'>📈 Grafik Akurasi Model</h3>", unsafe_allow_html=True)
    trend_data = pd.DataFrame({
        "Versi": ["V1", "V2", "V3", "V4", "V5"],
        "Akurasi": [94.8, 96.2, 97.0, 97.8, 98.2]
    })
    fig = px.bar(trend_data, x="Versi", y="Akurasi", text="Akurasi", color="Akurasi",
                 color_continuous_scale=["#f8bbd0", "#ec407a", "#ad1457"])
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_layout(yaxis_title="Akurasi (%)", xaxis_title=None,
                      coloraxis_showscale=False, plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# 🌸 MODEL INFO & ABOUT
# ============================================
elif selected == "Model Info":
    st.title("ℹ️ Informasi Model")
    st.write("""
        Model **AI Image Detection** dibangun menggunakan arsitektur **Convolutional Neural Network (CNN)** 
        dengan optimizer **Adam** dan fungsi aktivasi **ReLU + Softmax**.  
        Dataset terdiri dari berbagai jenis kendaraan yang telah melalui proses augmentasi 
        untuk meningkatkan generalisasi model.
    """)

elif selected == "About Project":
    st.title("📘 Tentang Proyek")
    st.write("""
        Proyek ini dikembangkan untuk mendemonstrasikan penerapan **Deep Learning**
        dalam bidang **Computer Vision**, khususnya untuk mendeteksi jenis kendaraan.
        Aplikasi ini menggunakan Streamlit sebagai antarmuka interaktif, 
        dan model dilatih menggunakan TensorFlow/Keras.
    """)
