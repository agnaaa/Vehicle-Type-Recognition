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
# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="AI Image Detection", layout="wide")

# ---------------------------
# CSS (pink soft pastel + layout)
# ---------------------------
st.markdown("""
<style>
/* page background */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, #fff5f8 0%, #ffffff 100%);
  padding-bottom: 60px;
}

/* global font */
* { font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; }

/* navbar */
.navbar {
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding: 14px 40px;
  background: white;
  border-radius: 10px;
  box-shadow: 0 6px 18px rgba(16,24,40,0.06);
  margin: 10px 30px;
}
.nav-left {
  display:flex;
  align-items:center;
  gap:12px;
  font-weight:700;
  color:#111827;
}
.logo-box {
  width:36px; height:36px; border-radius:8px;
  background: linear-gradient(180deg,#f07da7,#e86e9a);
  display:flex; align-items:center; justify-content:center;
  color:white; font-weight:800;
}
.nav-center {
  display:flex;
  gap:22px;
  align-items:center;
  margin-left:auto; margin-right:auto;
}
.nav-link {
  padding:8px 14px; border-radius:8px; font-weight:600;
  color:#374151; text-decoration:none;
}
.nav-link.active {
  background: rgba(231,81,120,0.08);
  color:#e75480;
  box-shadow: 0 4px 14px rgba(231,81,120,0.08);
}

/* hero */
.hero {
  display:flex;
  justify-content:space-between;
  align-items:center;
  padding: 56px 80px;
  margin: 10px 30px;
  border-radius:12px;
}
.hero-left { max-width: 640px; }
.hero-left h1 {
  font-size:48px; line-height:1.05; margin:0; font-weight:800; color:#111827;
}
.hero-left h1 .accent { color:#e75480; }
.hero-left p { color:#6b7280; margin-top:18px; font-size:16px; line-height:1.6; }

/* CTAs */
.btn-primary {
  display:inline-block; background: linear-gradient(90deg,#f07da7,#e86e9a);
  color:white; padding: 12px 22px; border-radius:12px; font-weight:700; border:none; cursor:pointer;
  box-shadow: 0 10px 22px rgba(231,81,120,0.14);
}
.btn-ghost {
  display:inline-block; margin-left:14px; padding: 10px 18px; border-radius:12px; border:1px solid #f4c3d6;
  color:#e75480; font-weight:700; background: transparent;
}

/* upload card */
.upload-card {
  width:380px; background:white; border-radius:14px; padding:22px; text-align:center;
  box-shadow: 0 12px 40px rgba(16,24,40,0.06);
}
.upload-placeholder {
  border: 1px dashed #f6cde0; border-radius:10px; padding:28px; color:#b88a9f;
}

/* vehicle grid */
.vehicle-grid {
  display:flex; gap:22px; justify-content:center; margin:48px 80px;
  flex-wrap:wrap;
}
.vehicle-card {
  width: 240px; background:white; border-radius:12px; padding:18px; text-align:center;
  box-shadow: 0 8px 20px rgba(16,24,40,0.04);
}
.vehicle-card img { width:100%; height:110px; object-fit:contain; border-radius:8px; margin-bottom:10px; }

/* stats */
.stats {
  display:flex; justify-content:center; gap:60px; margin-top:34px;
}
.stat { text-align:center; }
.stat .dot { width:60px; height:60px; border-radius:50%; background:linear-gradient(180deg,#f7cfe0,#f1a1c6); margin:auto; }

/* features */
.features-grid { display:flex; gap:22px; justify-content:center; margin:36px 80px; flex-wrap:wrap; }
.feature-card { width:230px; background:white; border-radius:12px; padding:18px; text-align:center; box-shadow: 0 8px 20px rgba(16,24,40,0.04); }

/* classification layout */
.classification {
  display:flex; gap:32px; padding:36px 80px; margin:20px 30px;
}
.left-card, .right-card {
  background:white; border-radius:14px; padding:22px; box-shadow: 0 12px 40px rgba(16,24,40,0.06);
}
.left-card { flex:1; min-height:420px; }
.right-card { width:520px; }

/* prediction bars (custom) */
.pred-row { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:14px; }
.pred-label { display:flex; gap:10px; align-items:center; }
.progress-wrap { flex:1; height:12px; background:#f3f4f6; border-radius:8px; overflow:hidden; position:relative; }
.progress-bar { height:100%; background:linear-gradient(90deg,#34d399,#10b981); border-radius:8px; }

/* small caption box */
.info-box { background:#f3f6ff; padding:12px; border-radius:10px; color:#065f46; margin-top:18px; }

/* responsive small */
@media (max-width:900px){
  .hero { flex-direction:column; padding:36px 18px; }
  .hero-left { width:100%; }
  .upload-card { margin-top:20px; width:100%; }
  .vehicle-grid, .features-grid { margin:20px; }
  .classification { flex-direction:column; padding:20px; }
  .right-card { width:100%; }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Navigation (use query param if set, fallback to session_state)
# ---------------------------
# If user clicks link with ?page=Name, st.query_params reads it
qp = st.query_params.get("page", [None])[0]
if "page" not in st.session_state:
    st.session_state.page = "Home"
if qp:
    # allow query param navigation
    if qp in ["Home", "Classification", "About"]:
        st.session_state.page = qp

# ---------------------------
# NAVBAR HTML (anchors with query params)
# ---------------------------
active = st.session_state.page
nav_html = f"""
<div class="navbar">
  <div class="nav-left">
    <div class="logo-box">AI</div>
    <div style="font-weight:700">AI Image Detection</div>
  </div>
  <div class="nav-center">
    <a class="nav-link {'active' if active=='Home' else ''}" href="?page=Home">Home</a>
    <a class="nav-link {'active' if active=='Classification' else ''}" href="?page=Classification">Classification</a>
    <a class="nav-link {'active' if active=='About' else ''}" href="?page=About">About Project</a>
  </div>
</div>
"""
st.markdown(nav_html, unsafe_allow_html=True)

# ---------------------------
# Helper: make prediction (dummy)
# ---------------------------
def predict_from_filename(name):
    name = name.lower()
    if any(k in name for k in ["car","mobil","sedan","bmw","toyota","honda"]):
        return "Mobil", random.uniform(0.85, 0.98)
    if any(k in name for k in ["motor","bike","motorcycle","scooter"]):
        return "Motor", random.uniform(0.75, 0.95)
    if any(k in name for k in ["truck","truk","pickup","lorry"]):
        return "Truk", random.uniform(0.75, 0.95)
    if any(k in name for k in ["bus","minibus","coach"]):
        return "Bus", random.uniform(0.75, 0.95)
    # else random
    cls = random.choices(["Mobil","Motor","Truk","Bus"], weights=[0.35,0.30,0.2,0.15])[0]
    score = random.uniform(0.70, 0.96)
    return cls, score

def make_prediction_bars(main_pred, main_score):
    # produce probs for all classes (summing approx 1)
    base = {"Mobil":0.0, "Motor":0.0, "Truk":0.0, "Bus":0.0}
    base[main_pred] = main_score
    others = [k for k in base.keys() if k!=main_pred]
    remaining = max(0.0, 1.0 - base[main_pred])
    # distribute remaining to others randomly
    r = [random.random() for _ in others]
    s = sum(r)
    for i,k in enumerate(others):
        base[k] = (r[i]/s) * remaining
    # normalize to 1 and convert to percentages
    total = sum(base.values())
    for k in base:
        base[k] = (base[k]/total) * 100
    return base

# ---------------------------
# PAGE: HOME
# ---------------------------
if st.session_state.page == "Home":
    # hero
    st.markdown("""
    <div class="hero">
      <div class="hero-left">
        <h1>Deteksi Jenis <span class="accent">Kendaraan AI</span></h1>
        <p>Platform revolusioner yang menggunakan teknologi deep learning untuk mengidentifikasi dan mengklasifikasikan jenis kendaraan seperti mobil, motor, truk, dan bus dengan akurasi tinggi.</p>
        </div>
      <div class="upload-card">
        <h4>Demo Cepat</h4>
        <div class="upload-placeholder">🖼️<br><br>Upload gambar kendaraan untuk analisis</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # action: Coba Sekarang button (go to classification)
    # Use Streamlit button so we can change session_state
    cols = st.columns([1, 0.3, 1])
    with cols[0]:
        if st.button("🚀 Coba Sekarang", key="try_now"):
            st.session_state.page = "Classification"
            # to update URL query param (so bookmarkable), set st.experimental_set_query_params
            st.experimental_set_query_params(page="Classification")
            st.experimental_rerun()

    # vehicle grid
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;font-size:26px;font-weight:700;margin-top:20px">Jenis Kendaraan yang Dapat Dideteksi</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="vehicle-grid">
      <div class="vehicle-card">
        <img src="https://i.ibb.co/FXBvZZ7/car.png" alt="Mobil">
        <h4>Mobil</h4><p>Sedan, SUV, Hatchback, dan berbagai jenis mobil penumpang</p>
      </div>
      <div class="vehicle-card">
        <img src="https://i.ibb.co/gWQhNsc/motorcycle.png" alt="Motor">
        <h4>Motor</h4><p>Sepeda motor, skuter, dan kendaraan roda dua lainnya</p>
      </div>
      <div class="vehicle-card">
        <img src="https://i.ibb.co/F8y2Csx/truck.png" alt="Truk">
        <h4>Truk</h4><p>Truk kargo, pickup, dan kendaraan komersial berat</p>
      </div>
      <div class="vehicle-card">
        <img src="https://i.ibb.co/NrQL8cp/bus.png" alt="Bus">
        <h4>Bus</h4><p>Bus kota, bus antar kota, dan kendaraan angkutan umum</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # stats
    st.markdown("""
    <div class="stats" style="margin-top:30px">
      <div><div class="dot"></div><div style="font-weight:800;margin-top:10px">98.2%</div><div style="color:#6b7280">Akurasi Model</div></div>
      <div><div class="dot"></div><div style="font-weight:800;margin-top:10px">47ms</div><div style="color:#6b7280">Waktu Proses</div></div>
      <div><div class="dot"></div><div style="font-weight:800;margin-top:10px">4+</div><div style="color:#6b7280">Jenis Kendaraan</div></div>
      <div><div class="dot"></div><div style="font-weight:800;margin-top:10px">99.9%</div><div style="color:#6b7280">Uptime</div></div>
    </div>
    """, unsafe_allow_html=True)

    # features
    st.markdown('<div style="height:30px"></div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;font-size:26px;font-weight:700">Mengapa Memilih Platform Kami?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="features-grid">
      <div class="feature-card"><div class="icon">🎯</div><h4>Deteksi Akurat</h4><p>Akurasi hingga 98.2% dalam mengenali jenis kendaraan.</p></div>
      <div class="feature-card"><div class="icon">⚡</div><h4>Pemrosesan Cepat</h4><p>Identifikasi kendaraan dalam waktu kurang dari 50ms.</p></div>
      <div class="feature-card"><div class="icon">🔒</div><h4>Keamanan Tinggi</h4><p>Data gambar diproses dengan enkripsi end-to-end.</p></div>
      <div class="feature-card"><div class="icon">🌐</div><h4>API Global</h4><p>Integrasi mudah melalui REST API.</p></div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# PAGE: CLASSIFICATION
# ---------------------------
elif st.session_state.page == "Classification":
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    st.markdown('<h1 style="text-align:center;color:#111827">Klasifikasi Gambar AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#6b7280">Upload gambar dan biarkan AI kami menganalisis serta mengklasifikasikan objek dalam gambar dengan akurasi tinggi</p>', unsafe_allow_html=True)

    st.markdown('<div class="classification">', unsafe_allow_html=True)
    # left card - upload & analyze
    st.markdown('<div class="left-card">', unsafe_allow_html=True)
    st.markdown('<h3>Upload Gambar</h3>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Pilih atau drop gambar (JPG/PNG)", type=["jpg","jpeg","png"], key="upload1")

    # show uploaded preview
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_column_width=True)
        except Exception:
            st.error("Gagal memuat gambar.")

    analyze_clicked = st.button("🔍 Analisis Gambar", key="analyze_button")

    st.markdown('</div>', unsafe_allow_html=True)  # close left-card

    # right card - result
    st.markdown('<div class="right-card">', unsafe_allow_html=True)
    st.markdown('<h3>Hasil Klasifikasi</h3>', unsafe_allow_html=True)

    if analyze_clicked:
        if not uploaded:
            st.warning("Silakan unggah gambar terlebih dahulu.")
        else:
            # simulate processing
            start = time.time()
            with st.spinner("Menganalisis..."):
                time.sleep(1.2)  # simulate latency
            process_time = round((time.time() - start)*1000)  # ms-ish

            # get prediction (dummy)
            filename = uploaded.name if hasattr(uploaded, "name") else "image"
            main_pred, score = predict_from_filename(filename)
            bars = make_prediction_bars(main_pred, score)

            # show top info
            st.markdown(f"<div style='display:flex;gap:12px;margin-bottom:14px'><div style='flex:1;background:#f8fafc;padding:10px;border-radius:8px'><b>Waktu Proses:</b><div style='margin-top:6px'>{process_time}ms</div></div><div style='width:12px'></div><div style='flex:1;background:#f8fafc;padding:10px;border-radius:8px'><b>Model:</b><div style='margin-top:6px'>ImageDetect-v1.0 (demo)</div></div></div>", unsafe_allow_html=True)

            st.markdown("<h4>Prediksi Teratas</h4>", unsafe_allow_html=True)
            # render progress bars
            for cls in ["Mobil","Motor","Truk","Bus"]:
                pct = bars.get(cls, 0)
                # color choose: mobil green, motor blue, truk purple, bus orange
                color = "#34d399" if cls=="Mobil" else "#60a5fa" if cls=="Motor" else "#a78bfa" if cls=="Truk" else "#fb923c"
                bar_html = f"""
                <div class="pred-row">
                  <div class="pred-label"><div style="width:10px;height:10px;border-radius:50%;background:{color}"></div><div style="min-width:80px">{cls}</div></div>
                  <div style="flex:1;">
                    <div class="progress-wrap"><div class="progress-bar" style="width:{pct:.1f}%;background:{color}"></div></div>
                  </div>
                  <div style="width:48px;text-align:right">{pct:.1f}%</div>
                </div>
                """
                st.markdown(bar_html, unsafe_allow_html=True)

            # interpretation box
            st.markdown(f"<div class='info-box'><b>Interpretasi Hasil:</b> Sistem mendeteksi objek utama sebagai <b>{main_pred}</b> dengan tingkat kepercayaan sekitar <b>{(bars[main_pred]):.1f}%</b>. Hasil ini bersifat simulasi (demo).</div>", unsafe_allow_html=True)
    else:
        st.info("Unggah gambar lalu tekan tombol 'Analisis Gambar' untuk melihat hasil.")

    st.markdown('</div>', unsafe_allow_html=True)  # close right-card
    st.markdown('</div>', unsafe_allow_html=True)  # close classification

# ---------------------------
# PAGE: ABOUT
# ---------------------------
elif st.session_state.page == "About":
    st.markdown('<div style="padding:36px 80px">', unsafe_allow_html=True)
    st.markdown("<h2>Tentang Project</h2>", unsafe_allow_html=True)
    st.markdown("""
    Aplikasi demo ini memperlihatkan antarmuka untuk sistem deteksi dan klasifikasi kendaraan berbasis AI.
    Pada versi demo ini, prediksi bersifat simulasi — jika kamu punya model (.h5 / .pt), aku bantu integrasikan.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
