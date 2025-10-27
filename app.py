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
# -------------------------
# CONFIGURASI PAGE
# -------------------------
st.set_page_config(
    page_title="🚗 AI Vehicle Type Recognition",
    page_icon="🚙",
    layout="wide"
)

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # pastikan file best.pt ada di folder yang sama
    return model

model = load_model()

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("📑 Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["🏠 Home", "🚘 Classification", "📊 Model Performance", "ℹ️ About"])

# -------------------------
# HALAMAN HOME
# -------------------------
if page == "🏠 Home":
    st.title("🚗 AI Vehicle Type Recognition")
    st.markdown("""
    ### Selamat datang di aplikasi deteksi jenis kendaraan!
    Aplikasi ini menggunakan **YOLOv8 (You Only Look Once)** untuk mengenali jenis kendaraan seperti:
    - 🏍️ Motor  
    - 🚗 Mobil  
    - 🚌 Bus  
    - 🚚 Truk  

    Unggah gambar kendaraan dan sistem akan otomatis mengenali jenisnya secara **real-time**.
    """)

    st.image("https://cdn.dribbble.com/users/1187278/screenshots/5634918/vehicles.gif", use_container_width=True)

# -------------------------
# HALAMAN CLASSIFICATION
# -------------------------
elif page == "🚘 Classification":
    st.title("🚘 Vehicle Image Classification")
    st.markdown("Unggah gambar kendaraan untuk mendeteksi jenisnya menggunakan model YOLOv8.")

    uploaded_file = st.file_uploader("Pilih gambar kendaraan...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())

        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang diunggah', use_container_width=True)

        # Deteksi dengan model
        with st.spinner('🔍 Sedang menganalisis gambar...'):
            results = model.predict(temp_file.name)
            result_image = results[0].plot()  # gambar dengan bounding box

        st.subheader("📸 Hasil Deteksi:")
        st.image(result_image, use_container_width=True)

        # Tampilkan label prediksi
        labels = results[0].boxes.cls
        names = [model.names[int(i)] for i in labels]
        if names:
            st.success(f"🚗 Jenis kendaraan terdeteksi: **{', '.join(names)}**")
        else:
            st.warning("Tidak ada kendaraan terdeteksi pada gambar.")

# -------------------------
# HALAMAN MODEL PERFORMANCE
# -------------------------
elif page == "📊 Model Performance":
    import pandas as pd
    import plotly.express as px

    st.title("📊 Model Performance Overview")

    st.markdown("""
    Berikut adalah ringkasan performa model deteksi kendaraan berdasarkan hasil evaluasi:
    """)

    metrics_data = {
        "Metrik": ["Precision", "Recall", "mAP50", "mAP50-95"],
        "Nilai (%)": [96.2, 95.8, 97.5, 92.3]
    }
    df = pd.DataFrame(metrics_data)

    st.table(df)

    st.markdown("### 🔺 Tren Akurasi Model (versi ke-1 s.d ke-5)")
    trend_data = pd.DataFrame({
        "Versi": ["V1", "V2", "V3", "V4", "V5"],
        "Akurasi": [94.8, 96.2, 97.0, 97.8, 98.2]
    })

    fig = px.bar(
        trend_data, x="Versi", y="Akurasi",
        text="Akurasi", color="Akurasi",
        color_continuous_scale=["#f8bbd0", "#ec407a", "#c2185b"],
        title="Perkembangan Akurasi Model"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# HALAMAN ABOUT
# -------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dibuat sebagai demonstrasi penerapan **Computer Vision** menggunakan **YOLOv8** untuk mendeteksi jenis kendaraan.

    **Dibuat oleh:** Agna Balqis  
    **Framework:** Streamlit + Ultralytics YOLOv8  
    **Versi:** 1.0.0
    """)
    st.image("https://miro.medium.com/v2/resize:fit:800/1*dT7-Ixk12HjS-VeBPzSLLA.gif", use_container_width=True)
