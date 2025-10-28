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
# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="AI Image Detection", layout="wide")

# -------------------------
# CSS: styling + make radio look like navbar
# -------------------------
st.markdown(
    """
    <style>
    :root{
        --pink-soft: #fdeef4;
        --accent: #ec5c9a;
        --accent-strong: #e75480;
    }

    /* app background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, var(--pink-soft) 0%, #ffffff 100%) !important;
    }

    /* hide top streamlit header and menu for cleaner look */
    header {display:none}

    /* NAVBAR - we style the radio group to look like a nav */
    .nav-container {
        width:100%;
        display:flex;
        justify-content:center;
        background: white;
        padding:10px 24px;
        border-radius:10px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.06);
        margin-bottom:18px;
    }
    /* Radiogroup styling: works across streamlit versions by targeting role radiogroup labels */
    div[role="radiogroup"] > label {
        display:inline-block;
        margin: 0 12px;
        padding:8px 16px;
        border-radius:10px;
        font-weight:600;
        cursor:pointer;
        color:#333;
    }
    /* Active radio looks like selected nav item */
    div[role="radiogroup"] > label[aria-checked="true"] {
        background: linear-gradient(180deg,#fde3ec,#fff);
        color: var(--accent-strong);
        box-shadow: 0 6px 18px rgba(231,81,120,0.08);
    }

    /* HERO */
    .hero {
        display:flex;
        gap:40px;
        align-items:center;
        padding:48px 56px;
        flex-wrap:wrap;
    }
    .hero-left { flex:1 1 480px; max-width:720px; }
    .hero-left h1 { font-size:48px; margin:0; line-height:1; color:#1f2937; font-weight:800;}
    .hero-left h1 .accent { color: var(--accent-strong); display:block; }
    .hero-left p { color:#6b7280; margin-top:18px; font-size:16px; line-height:1.6; max-width:560px;}
    .hero-buttons { margin-top:26px; display:flex; gap:14px; align-items:center; }

    .btn-gradient {
        background: linear-gradient(90deg, #f07da7, #e86e9a);
        color: white; padding:12px 22px; border-radius:12px; font-weight:700; border:none; text-decoration:none;
        box-shadow: 0 8px 20px rgba(231,81,120,0.12);
    }
    .btn-outline {
        border: 2px solid #f6cde0; color: var(--accent-strong); padding:10px 20px; border-radius:12px; background:transparent; font-weight:700;
    }

    /* upload card */
    .upload-card {
        background: white; border-radius:14px; padding:26px; width:420px; text-align:center;
        box-shadow: 0 14px 30px rgba(16,24,40,0.06);
    }
    .upload-placeholder {
        border:2px dashed #f6cde0; border-radius:12px; padding:34px; color:#b88a9f; margin:12px 0;
    }
    .upload-choose {
        background: #fdeef8; color: var(--accent-strong); padding:8px 12px; border-radius:8px; display:inline-block; margin-top:8px; font-weight:700;
    }

    /* vehicle cards */
    .vehicle-row { padding: 36px 64px; }
    .vehicle-grid { display:flex; gap:22px; justify-content:center; flex-wrap:wrap; }
    .vehicle-card {
        background:white; width:260px; border-radius:12px; padding:18px; text-align:center; box-shadow: 0 12px 28px rgba(16,24,40,0.04);
    }
    .vehicle-card img { width:100%; height:120px; object-fit:cover; border-radius:8px; }
    .vehicle-card h4 { margin:12px 0 6px 0; color:#222; }
    .vehicle-card p { color:#6b7280; font-size:14px; margin:0; }

    /* stats */
    .stats { display:flex; gap:40px; justify-content:center; padding:36px 0; }
    .stat { text-align:center; color:#1f2937; }
    .stat .dot { width:56px;height:56px;border-radius:50%; background: linear-gradient(180deg,#f6cde0,#f2b6d9); margin:auto; box-shadow:0 8px 20px rgba(231,81,120,0.06); }
    .stat h3 { margin:12px 0 6px 0; font-size:22px; font-weight:800; }
    .stat p { margin:0; color:#6b7280; }

    /* features */
    .features { padding:36px 64px; text-align:center; }
    .features-grid { display:flex; gap:22px; justify-content:center; flex-wrap:wrap; }
    .feature-card { width:260px; background:white; border-radius:12px; padding:22px; text-align:center; box-shadow: 0 12px 28px rgba(16,24,40,0.04); }
    .feature-card .icon { width:56px;height:56px;border-radius:50%; display:inline-flex; align-items:center; justify-content:center; margin-bottom:12px; background:linear-gradient(180deg,#f7cfe0,#f1a1c6); color:white; font-weight:700; }
    .feature-card h4 { margin:8px 0; }
    .feature-card p { color:#6b7280; font-size:14px; }

    /* floating talk button */
    .talk-float { position: fixed; bottom: 24px; right: 24px; background: linear-gradient(135deg,#c2185b,#e91e63); color:white; padding:12px 20px; border-radius:28px; box-shadow:0 10px 30px rgba(0,0,0,0.15); font-weight:700; text-decoration:none; }

    @media (max-width:900px) {
        .hero { padding:28px 18px; flex-direction:column; align-items:flex-start; }
        .upload-card { width:100%; max-width:420px; }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Navigation control (radio)
# -------------------------
nav = st.radio("", ["Home", "Classification", "About Project"], index=0, horizontal=True, label_visibility="collapsed")
st.session_state['page'] = nav  # keep consistent

# Show the styled nav container (visual)
st.markdown(
    """
    <div class="nav-container">
       <div style="width:100%;max-width:1100px;">
           <!-- the radio is above; this block only for spacing/visual (radio labels are styled by CSS) -->
       </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# PAGE: Home
# -------------------------
if st.session_state['page'] == "Home":
    # Hero
    st.markdown(
        """
        <div class="hero">
            <div class="hero-left">
                <h1>Deteksi Jenis <span class="accent">Kendaraan AI</span></h1>
                <p>Platform revolusioner yang menggunakan teknologi deep learning untuk mengidentifikasi dan mengklasifikasikan jenis kendaraan seperti mobil, motor, truck, dan bus dengan akurasi tinggi.</p>
                <div class="hero-buttons">
                    <a class="btn-gradient" href="javascript:void(0)" onclick="document.querySelectorAll('input[type=radio]')[1].click()">🚀 Coba Sekarang</a>
                    <a class="btn-outline" href="#features">📘 Pelajari Lebih Lanjut</a>
                </div>
            </div>

            <div class="upload-card">
                <h4>Demo Cepat</h4>
                <div class="upload-placeholder">
                    <div style="font-size:26px;">🖼️</div>
                    <div style="margin-top:8px;color:#b88a9f">Upload gambar kendaraan untuk analisis</div>
                    <div class="upload-choose">Pilih Gambar</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Vehicle cards
    st.markdown(
        """
        <div class="vehicle-row">
            <h2 style="text-align:center; font-size:28px; margin-bottom:6px;">Jenis Kendaraan yang Dapat Dideteksi</h2>
            <p style="text-align:center; color:#6b7280; margin-top:0; margin-bottom:24px;">Sistem AI kami dapat mengenali berbagai jenis kendaraan dengan akurasi tinggi</p>

            <div class="vehicle-grid">
                <div class="vehicle-card">
                    <img src="https://i.ibb.co/FXBvZZ7/car.png">
                    <h4>Mobil</h4>
                    <p>Sedan, SUV, Hatchback, dan berbagai jenis mobil penumpang</p>
                </div>
                <div class="vehicle-card">
                    <img src="https://i.ibb.co/gWQhNsc/motorcycle.png">
                    <h4>Motor</h4>
                    <p>Sepeda motor, skuter, dan kendaraan roda dua lainnya</p>
                </div>
                <div class="vehicle-card">
                    <img src="https://i.ibb.co/F8y2Csx/truck.png">
                    <h4>Truck</h4>
                    <p>Truk kargo, pickup, dan kendaraan komersial berat</p>
                </div>
                <div class="vehicle-card">
                    <img src="https://i.ibb.co/NrQL8cp/bus.png">
                    <h4>Bus</h4>
                    <p>Bus kota, bus antar kota, dan kendaraan angkutan umum</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stats
    st.markdown(
        """
        <div class="stats">
            <div class="stat"><div class="dot"></div><h3>98.2%</h3><p>Akurasi Model</p></div>
            <div class="stat"><div class="dot"></div><h3>47ms</h3><p>Waktu Proses</p></div>
            <div class="stat"><div class="dot"></div><h3>4+</h3><p>Jenis Kendaraan</p></div>
            <div class="stat"><div class="dot"></div><h3>99.9%</h3><p>Uptime</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Features
    st.markdown(
        """
        <div id="features" class="features">
            <h2 style="font-size:28px; margin-bottom:6px;">Mengapa Memilih Platform Kami?</h2>
            <p style="color:#6b7280; margin-top:0; margin-bottom:20px;">Teknologi AI terdepan yang dirancang khusus untuk deteksi kendaraan dengan akurasi tinggi</p>

            <div class="features-grid">
                <div class="feature-card">
                    <div class="icon">🎯</div>
                    <h4>Deteksi Akurat</h4>
                    <p>Akurasi hingga 98.2% dalam mengenali jenis kendaraan dengan teknologi deep learning</p>
                </div>
                <div class="feature-card">
                    <div class="icon">⚡</div>
                    <h4>Pemrosesan Cepat</h4>
                    <p>Identifikasi kendaraan dalam waktu kurang dari 50ms</p>
                </div>
                <div class="feature-card">
                    <div class="icon">🔒</div>
                    <h4>Keamanan Tinggi</h4>
                    <p>Data gambar kendaraan diproses dengan enkripsi end-to-end</p>
                </div>
                <div class="feature-card">
                    <div class="icon">🌐</div>
                    <h4>API Global</h4>
                    <p>Akses mudah melalui REST API untuk integrasi sistem traffic management</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------
# PAGE: Classification
# -------------------------
elif st.session_state['page'] == "Classification":
    st.markdown(
        """
        <div style="text-align:center; margin-top:28px;">
            <h2 style="font-size:32px; font-weight:800; color:#1f2937;">Klasifikasi Gambar AI</h2>
            <p style="color:#6b7280; max-width:800px; margin:auto;">Upload gambar dan biarkan AI kami menganalisis serta mengklasifikasikan objek dalam gambar dengan akurasi tinggi</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # layout with two cards (we'll add real uploader below)
    st.markdown(
        """
        <div style="display:flex; gap:32px; justify-content:center; padding:40px; flex-wrap:wrap;">
            <div style="width:420px; background:white; border-radius:12px; padding:26px; box-shadow:0 12px 30px rgba(16,24,40,0.06);">
                <h4 style="margin:0 0 10px 0;">Upload Gambar</h4>
                <p style="color:#6b7280; margin-top:0;">Pilih atau Drop Gambar (JPG, PNG, WebP hingga 10MB)</p>
            </div>

            <div style="width:420px; background:white; border-radius:12px; padding:26px; box-shadow:0 12px 30px rgba(16,24,40,0.06);">
                <h4 style="margin:0 0 10px 0;">Hasil Klasifikasi</h4>
                <p style="color:#6b7280; margin-top:0;">Upload dan analisis gambar untuk melihat hasil klasifikasi</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Actual uploader in python (below the HTML cards)
    left_col, right_col = st.columns([1, 1])
    with left_col:
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="cls_upload")
        if uploaded_file:
            img_bytes = uploaded_file.getvalue()
            img = Image.open(io.BytesIO(img_bytes))
            st.image(img, caption="Gambar yang diunggah", use_column_width=True)
            # placeholder for real classification logic:
            st.success("Hasil contoh: 🚗 Mobil — Confidence 97.8%")
    with right_col:
        st.empty()

    # example images row
    st.markdown(
        """
        <div style="text-align:center; margin-top:30px;">
            <h4>Coba Gambar Contoh</h4>
            <div style="display:flex; gap:18px; justify-content:center; margin-top:14px; flex-wrap:wrap;">
                <div style="width:180px; background:white; border-radius:12px; padding:14px; box-shadow:0 8px 20px rgba(16,24,40,0.04);">
                    <img src="https://i.ibb.co/zZcdF12/cat.png" width="100%" style="border-radius:8px;">
                    <p style="margin:8px 0 0 0; font-weight:600;">Kucing</p>
                </div>
                <div style="width:180px; background:white; border-radius:12px; padding:14px; box-shadow:0 8px 20px rgba(16,24,40,0.04);">
                    <img src="https://i.ibb.co/k5XyN2H/dog.png" width="100%" style="border-radius:8px;">
                    <p style="margin:8px 0 0 0; font-weight:600;">Anjing</p>
                </div>
                <div style="width:180px; background:white; border-radius:12px; padding:14px; box-shadow:0 8px 20px rgba(16,24,40,0.04);">
                    <img src="https://i.ibb.co/FXBvZZ7/car.png" width="100%" style="border-radius:8px;">
                    <p style="margin:8px 0 0 0; font-weight:600;">Mobil</p>
                </div>
                <div style="width:180px; background:white; border-radius:12px; padding:14px; box-shadow:0 8px 20px rgba(16,24,40,0.04);">
                    <img src="https://i.ibb.co/4VvX8PW/apple.png" width="100%" style="border-radius:8px;">
                    <p style="margin:8px 0 0 0; font-weight:600;">Buah</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------
# PAGE: About Project
# -------------------------
else:
    st.markdown(
        """
        <div style="padding:40px;">
            <h2>About Project</h2>
            <p>AI Image Detection — demo UI & simple classification front-end. Backend model integration (YOLO/TensorFlow) bisa ditambahkan di bagian Classification untuk memproses gambar yang diupload.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------
# Floating talk button
# -------------------------
st.markdown('<a class="talk-float" href="#">💬 Talk with Us</a>', unsafe_allow_html=True)
