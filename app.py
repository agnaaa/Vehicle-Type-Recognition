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
import base64
from pathlib import Path


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
from PIL import Image
from ultralytics import YOLO
import webbrowser
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile, os

st.set_page_config(page_title="Vehicle Recognition App", layout="wide")

# --- Navigation State ---
if "page" not in st.session_state:
    st.session_state.page = "Home"

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("🚗 Navigasi")
    menu = st.radio(
        "Pilih Halaman",
        ["Home", "Classification"],
        index=0 if st.session_state.page == "Home" else 1,
    )
    st.session_state.page = menu

# ===============================
# 🏠 HOME PAGE
# ===============================
if st.session_state.page == "Home":
    st.markdown("<h1 style='text-align:center;'>🚘 Sistem Deteksi Jenis Kendaraan</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='text-align:justify; font-size:17px;'>
        Aplikasi ini menggunakan model kecerdasan buatan berbasis <b>YOLOv8</b> untuk mendeteksi
        dan mengklasifikasikan jenis kendaraan secara otomatis dari gambar yang diunggah.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.image("images/Image_60.png", caption="Contoh deteksi kendaraan", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Coba Sekarang", use_container_width=True, type="primary"):
        # langsung ubah halaman ke Classification
        st.session_state.page = "Classification"
        st.rerun()

# ===============================
# 🔍 CLASSIFICATION PAGE
# ===============================
elif st.session_state.page == "Classification":
    st.markdown("<h2 style='text-align:center;'>🔍 Klasifikasi Jenis Kendaraan</h2>", unsafe_allow_html=True)
    left, right = st.columns([1, 0.8])

    with left:
        uploaded = st.file_uploader("Unggah gambar kendaraan", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Gambar yang diunggah", use_container_width=True)

    with right:
        if uploaded:
            try:
                tmp_dir = tempfile.mkdtemp()
                tmp_path = os.path.join(tmp_dir, uploaded.name)
                img.save(tmp_path)

                model = YOLO("model/best.pt")  # model hasil training kamu
                results = model.predict(tmp_path, conf=0.4)

                for r in results:
                    result_img = r.plot()  # gambar hasil deteksi
                    st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

                    detected_classes = [model.names[int(c)] for c in r.boxes.cls]
                    if detected_classes:
                        st.success(f"🚗 Jenis Kendaraan Terdeteksi: {', '.join(set(detected_classes))}")
                    else:
                        st.warning("⚠️ Tidak ada kendaraan terdeteksi dalam gambar ini.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
        else:
            st.info("Unggah gambar kendaraan untuk memulai deteksi.")
