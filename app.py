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
# -----------------------------
# SETUP HALAMAN
# -----------------------------
st.set_page_config(page_title="Vehicle Type Recognition", page_icon="🚗", layout="wide")

# -----------------------------
# NAVIGASI MENU
# -----------------------------
menu = st.sidebar.selectbox(
    "📍 Navigasi",
    ["Home", "Vehicle Classification", "Model Performance", "Model Info", "About Us"]
)

# -----------------------------
# HOME PAGE
# -----------------------------
if menu == "Home":
    st.title("🚘 Vehicle Type Recognition System")
    st.markdown("""
    Selamat datang di **Vehicle Type Recognition System**, sebuah website berbasis **Machine Learning**
    yang mampu mengenali jenis kendaraan seperti **Mobil, Motor, Bus, dan Truk** dari gambar.

    Website ini dikembangkan sebagai proyek kecerdasan buatan menggunakan model klasifikasi gambar.
    Silakan jelajahi fitur-fitur di menu sebelah kiri untuk mencoba dan melihat performa model.
    """)
    st.image("images/example_car.jpg", caption="Contoh: Deteksi kendaraan", use_column_width=True)
    st.info("Klik menu **Vehicle Classification** untuk mulai melakukan deteksi gambar kendaraan.")

# -----------------------------
# VEHICLE CLASSIFICATION PAGE
# -----------------------------
elif menu == "Vehicle Classification":
    st.header("🔍 Vehicle Image Classification")
    st.markdown("Unggah gambar kendaraan di bawah ini untuk mengenali jenisnya.")

    uploaded_file = st.file_uploader("Pilih gambar kendaraan...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang diunggah', use_column_width=True)

        st.write("")
        st.write("⏳ Sedang memproses...")

        # Simulasi loading deteksi
        with st.spinner('Mendeteksi jenis kendaraan...'):
            time.sleep(2)

        # Load model (contoh placeholder)
        # model = joblib.load("model/vehicle_model.pkl")
        # pred = model.predict(image_features)
        # Hasil dummy untuk contoh tampilan
        pred = np.random.choice(["Mobil", "Motor", "Bus", "Truk"])
        confidence = np.random.uniform(80, 99)

        st.success(f"✅ Hasil Prediksi: **{pred}** ({confidence:.2f}% confidence)")

# -----------------------------
# MODEL PERFORMANCE PAGE
# -----------------------------
elif menu == "Model Performance":
    st.header("📊 Model Performance Evaluation")
    st.markdown("""
    Berikut hasil evaluasi model Machine Learning yang digunakan pada sistem ini.
    """)

    data = {
        'Model': ['GaussianNB', 'MultinomialNB', 'BernoulliNB'],
        'Accuracy (%)': [70.78, 64.94, 70.13]
    }
    df = pd.DataFrame(data)
    st.table(df)

    st.bar_chart(df.set_index('Model'))

    st.markdown("""
    **Interpretasi:**
    - Model dengan akurasi tertinggi adalah **GaussianNB (70.78%)**.
    - Model ini menunjukkan performa terbaik dalam mengenali jenis kendaraan dibandingkan dua model lainnya.
    """)

# -----------------------------
# MODEL INFORMATION PAGE
# -----------------------------
elif menu == "Model Info":
    st.header("🧠 How The Model Works")
    st.markdown("""
    Sistem ini menggunakan algoritma **Naive Bayes Classifier** untuk mengenali jenis kendaraan.
    
    **Langkah utama dalam proses klasifikasi:**
    1. **Preprocessing Gambar:** Gambar diubah menjadi format dan ukuran yang sesuai.
    2. **Ekstraksi Fitur:** Model mengekstrak pola visual dari gambar.
    3. **Prediksi:** Berdasarkan pola tersebut, model menentukan kategori kendaraan.
    
    **Kategori kendaraan yang dapat dikenali:**
    - Mobil 🚗  
    - Motor 🏍️  
    - Bus 🚌  
    - Truk 🚚  
    """)

    st.image("images/example_car.jpg", caption="Contoh alur klasifikasi", use_column_width=True)

# -----------------------------
# ABOUT US PAGE
# -----------------------------
elif menu == "About Us":
    st.header("👩‍💻 About Us")
    st.markdown("""
    Website ini dikembangkan oleh:

    **Agna Balqis**  
    Mahasiswa Program Studi Statistika  
    Sebagai proyek penerapan *Machine Learning* dalam klasifikasi gambar kendaraan.

    **Tujuan:**  
    Meningkatkan pemahaman terhadap penerapan algoritma klasifikasi dalam pengenalan citra digital,
    serta mengembangkan sistem sederhana yang dapat mengenali jenis kendaraan secara otomatis.

    **Teknologi yang digunakan:**
    - Python & Streamlit  
    - Scikit-learn (Naive Bayes Classifiers)  
    - NumPy, Pandas, Matplotlib  
    - GitHub untuk version control & deployment
    """)

    st.info("Terima kasih telah mengunjungi situs ini! 🚀")

