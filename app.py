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
# ==========================
# ---------- Page config ----------
st.set_page_config(page_title="AI Vehicle Detection", page_icon="🚗", layout="wide")

# ---------- Custom CSS ----------
st.markdown(
    """
    <style>
    :root{
      --bg: #fbf7fb;
      --card: #ffffff;
      --accent: #f6c7e1; /* pink pastel */
      --accent-2: #f1e6f6; /* soft purple */
      --text: #222;
    }
    body { background-color: var(--bg); }
    .hero {
      padding: 56px 48px;
      border-radius: 14px;
      background: linear-gradient(90deg, rgba(255,255,255,0.9), rgba(255,255,255,0.85));
      box-shadow: 0 12px 30px rgba(0,0,0,0.08);
    }
    .hero-title { font-size: 48px; font-weight:700; color: #111; margin:0; line-height:1.05; }
    .hero-sub { font-size: 40px; font-weight:700; color: #e56fb1; margin:4px 0 18px 0; }
    .hero-desc { color: #444; font-size:16px; margin-bottom:18px; }
    .cta-primary {
      background: linear-gradient(90deg,#ff9fc7,#e56fb1);
      color: white;
      padding: 12px 22px;
      border-radius: 10px;
      font-weight:600;
      border: none;
      box-shadow: 0 6px 18px rgba(229,111,177,0.25);
    }
    .cta-secondary {
      border: 2px solid #f3d5e4;
      color: #c85b9a;
      padding: 10px 18px;
      border-radius: 10px;
      background: transparent;
      font-weight:600;
      margin-left:12px;
    }
    .upload-card {
      background: var(--card);
      border-radius: 12px;
      padding: 26px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.06);
      text-align:center;
    }
    .upload-drop {
      border: 2px dashed #f3d5e4;
      padding: 28px;
      border-radius: 10px;
      color: #c88ca9;
      margin-bottom: 14px;
      background: linear-gradient(180deg, rgba(246,199,225,0.03), rgba(241,230,246,0.02));
    }
    .small-muted { color:#777; font-size:13px; }
    .logo { height:36px; }
    /* responsive tweak */
    @media (max-width: 900px) {
      .hero-title { font-size: 34px; }
      .hero-sub { font-size: 30px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Top navigation (simple) ----------
col1, col2 = st.columns([1, 3])
with col1:
    # replace with your small logo file if present
    if os.path.exists("images/logo.png"):
        st.image("images/logo.png", width=120)
    else:
        st.markdown("<h3 style='margin:0; color:#c85b9a;'>AI Image Detection</h3>", unsafe_allow_html=True)
with col2:
    # simple nav line (visual only)
    st.markdown(
        "<div style='text-align:right; margin-top:8px;'>"
        "<a href='#' style='margin-right:18px; color:#666; text-decoration:none;'>Home</a>"
        "<a href='#' style='margin-right:18px; color:#666; text-decoration:none;'>Classification</a>"
        "<a href='#' style='margin-right:18px; color:#666; text-decoration:none;'>Model Performance</a>"
        "<a href='#' style='color:#666; text-decoration:none;'>About Us</a>"
        "</div>",
        unsafe_allow_html=True,
    )

st.write("")  # spacer

# ---------- Hero section ----------
with st.container():
    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    left, right = st.columns([2, 1.2])

    # LEFT: title, subtitle, description, CTAs
    with left:
        st.markdown("<h1 class='hero-title'>Deteksi Jenis</h1>", unsafe_allow_html=True)
        st.markdown("<h1 class='hero-sub'>Kendaraan AI</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p class='hero-desc'>Platform revolusioner yang menggunakan teknologi deep learning "
            "untuk mengidentifikasi dan mengklasifikasikan jenis kendaraan seperti mobil, motor, "
            "truck, dan bus dengan akurasi tinggi.</p>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([0.5, 0.5])
        with c1:
            # Primary CTA - jump to classification (we simulate with anchor id)
            if st.button("Coba Sekarang", key="cta_try"):
                st.experimental_rerun()  # in full app you would navigate to page
        with c2:
            st.markdown("<button class='cta-secondary'>Pelajari Lebih Lanjut</button>", unsafe_allow_html=True)

    # RIGHT: upload demo card
    with right:
        st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0;'>Demo Cepat</h3>", unsafe_allow_html=True)
        st.markdown("<div class='upload-drop'>Upload gambar kendaraan untuk analisis</div>", unsafe_allow_html=True)

        uploaded = st.file_uploader("Pilih Gambar", type=["png", "jpg", "jpeg"], key="hero_uploader")
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Preview", use_column_width=True)
            with st.spinner("Menganalisis gambar..."):
                time.sleep(1.5)
                # Placeholder result — replace with model inference
                import random

                pred = random.choice(["Mobil", "Motor", "Bus", "Truk"])
                conf = random.uniform(85, 99)
            st.success(f"Hasil Prediksi: **{pred}** — Confidence: {conf:.1f}%")
        else:
            # try to show local example image (optional)
            sample_path = "images/example_car.jpg"
            if os.path.exists(sample_path):
                sample = Image.open(sample_path)
                st.image(sample, caption="Contoh: Deteksi kendaraan", use_column_width=True)
            else:
                st.markdown("<p class='small-muted'>Tidak ada gambar contoh. Upload file di atas untuk mencoba.</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Optional: space for more content ----------
st.write("") 
st.write("") 
st.write("") 

# (Below this section you can add more content: features, how-it-works, team, footer...)

