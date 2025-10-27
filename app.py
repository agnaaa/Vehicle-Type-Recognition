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
# -------------------------
# Utility functions
# -------------------------
def pil_to_bytes(img: Image.Image) -> bytes:
    b = BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()

def draw_boxes_on_pil(img: Image.Image, boxes, labels=None, scores=None, color=(240,125,167)):
    """
    boxes: list of [x1, y1, x2, y2]
    labels: list of str
    scores: list of float
    """
    draw = ImageDraw.Draw(img)
    try:
        # try using a ttf font if available
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=14)
    except Exception:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        x1,y1,x2,y2 = box
        draw.rectangle([x1,y1,x2,y2], outline=color, width=3)
        caption = ""
        if labels:
            caption += str(labels[i])
        if scores:
            caption += f" {scores[i]:.2f}"
        if caption:
            text_w, text_h = draw.textsize(caption, font=font)
            # background rectangle for text
            draw.rectangle([x1, y1-text_h-6, x1+text_w+8, y1], fill=(255,255,255))
            draw.text((x1+4, y1-text_h-4), caption, fill=(0,0,0), font=font)
    return img

def preprocess_for_classifier(crop: Image.Image, target_size):
    """Resize and normalize crop image for classifier (float32 [0,1])."""
    img = crop.convert("RGB")
    img = img.resize((target_size[1], target_size[0]))  # target_size = (h,w,c)
    arr = np.array(img).astype("float32") / 255.0
    # ensure shape (h,w,c)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    return np.expand_dims(arr, axis=0)  # (1,h,w,c)

# -------------------------
# Load models (cached)
# -------------------------
@st.cache_resource
def load_models():
    # load YOLO
    yolo_model = None
    classifier = None
    yolo_path = "best.pt"
    clf_path = "model/classifier_model.h5"

    # load YOLO (ultralytics)
    if YOLO is None:
        st.warning("Module ultralytics belum terinstal — YOLO tidak tersedia.")
    else:
        if os.path.exists(yolo_path):
            try:
                yolo_model = YOLO(yolo_path)
            except Exception as e:
                st.warning(f"⚠️ Gagal memuat YOLO model: {e}")
                yolo_model = None
        else:
            st.info("YOLO weights (best.pt) tidak ditemukan di folder project. Upload atau letakkan file best.pt jika mau deteksi.")

    # load classifier (optional)
    if tf is None:
        st.warning("Module tensorflow belum terinstal — classifier tidak tersedia.")
    else:
        if os.path.exists(clf_path):
            try:
                # compile=False for safety
                classifier = tf.keras.models.load_model(clf_path, compile=False)

                # normalize to single-input model if multiple inputs present
                if isinstance(classifier.input, list) and len(classifier.input) > 1:
                    st.warning("⚠️ Classifier model punya lebih dari satu input. Hanya input pertama yang akan digunakan.")
                    input_layer = classifier.input[0]
                    output_layer = classifier.output
                    classifier = tf.keras.Model(inputs=input_layer, outputs=output_layer)
            except Exception as e:
                st.warning(f"⚠️ Gagal memuat classifier model: {e}")
                classifier = None
        else:
            st.info("Classifier model (model/classifier_model.h5) tidak ditemukan. Klasifikasi halaman akan dinonaktifkan.")

    return yolo_model, classifier

# -------------------------
# Inference helpers
# -------------------------
def run_yolo_inference(yolo_model, pil_image, imgsz=640, conf=0.25):
    """
    Return: boxes (list of [x1,y1,x2,y2]), scores, class_ids, class_names
    Uses ultralytics YOLO model
    """
    if yolo_model is None:
        return [], [], [], []

    # ultralytics accepts np array (H,W,C) RGB
    np_img = np.array(pil_image.convert("RGB"))
    results = yolo_model.predict(source=np_img, imgsz=imgsz, conf=conf, verbose=False)

    # results is list of Results for each image; we have 1
    boxes, scores, class_ids, class_names = [], [], [], []
    try:
        r = results[0]
        if hasattr(r, "boxes") and r.boxes is not None:
            for b in r.boxes:
                # b.xyxy, b.conf, b.cls
                xyxy = b.xyxy.cpu().numpy().tolist()  # [[x1,y1,x2,y2]]
                if len(xyxy) > 0:
                    x1,y1,x2,y2 = xyxy[0]
                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    scores.append(float(b.conf.cpu().numpy()))
                    cls_id = int(b.cls.cpu().numpy())
                    class_ids.append(cls_id)
                    # try get name from model.names
                    name = r.names.get(cls_id, str(cls_id)) if hasattr(r, "names") else str(cls_id)
                    class_names.append(name)
    except Exception as e:
        st.warning(f"⚠️ Error saat memproses hasil YOLO: {e}")
    return boxes, scores, class_ids, class_names

def run_classifier_on_crops(classifier, pil_image, boxes):
    """
    For each box crop from pil_image, run classifier and return label/prob.
    classifier should be a tf.keras.Model
    """
    if classifier is None:
        return [None] * len(boxes), [None] * len(boxes)

    # determine classifier input shape: (None, h, w, c)
    try:
        input_shape = classifier.input_shape  # could be (None, h, w, c)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        # ensure shape length 4
        if len(input_shape) == 4:
            target_hw = (input_shape[1], input_shape[2])
            c = input_shape[3] if input_shape[3] is not None else 3
            target_size = (target_hw[0], target_hw[1], c)
        else:
            # fallback
            target_size = (224, 224, 3)
    except Exception:
        target_size = (224, 224, 3)

    pred_labels = []
    pred_probs = []
    for box in boxes:
        x1,y1,x2,y2 = box
        crop = pil_image.crop((x1, y1, x2, y2))
        arr = preprocess_for_classifier(crop, target_size)
        try:
            preds = classifier.predict(arr)
            # if output is logits/probs
            if preds.ndim == 2 and preds.shape[1] > 1:
                idx = int(np.argmax(preds, axis=1)[0])
                prob = float(np.max(tf.nn.softmax(preds, axis=1)))
                label = str(idx)
                # try to see if classifier has class names attribute (not guaranteed)
                if hasattr(classifier, "class_names"):
                    label = classifier.class_names[idx]
            else:
                # single output (regression or binary)
                val = float(preds.flatten()[0])
                label = f"{val:.3f}"
                prob = val
            pred_labels.append(label)
            pred_probs.append(prob)
        except Exception as e:
            pred_labels.append(None)
            pred_probs.append(None)
            st.warning(f"⚠️ Error saat prediksi classifier: {e}")

    return pred_labels, pred_probs

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Image Detection", page_icon="🛻", layout="wide")

# Load models once (cached)
yolo_model, classifier = load_models()

# CSS + Hero styling (mirip desain)
st.markdown("""
    <style>
    :root{
      --pink-500:#f07da7;
      --muted:#6b7280;
    }
    .hero {
      display:flex; gap:40px; align-items:center; padding:40px 0;
    }
    .card {
      background: white;
      border-radius: 12px;
      box-shadow: 0 8px 20px rgba(16,24,40,0.06);
      padding: 20px;
    }
    .btn-primary {
      background: linear-gradient(90deg,var(--pink-500),#e86e9a);
      color: white; padding:10px 18px; border-radius:10px; font-weight:600;
      border:none;
    }
    .btn-ghost {
      background:transparent; border:1px solid #f6cde0; color:var(--pink-500);
      padding:10px 18px; border-radius:10px; font-weight:600;
    }
    </style>
""", unsafe_allow_html=True)

# Header / Navigation (simple)
colh1, colh2 = st.columns([1,3])
with colh1:
    st.markdown("<div style='display:flex;align-items:center;gap:10px'><div style='width:44px;height:44px;border-radius:8px;background:#f7c5d8;display:flex;align-items:center;justify-content:center;font-weight:700'>AI</div><strong>AI Image Detection</strong></div>", unsafe_allow_html=True)
with colh2:
    st.markdown("<div style='text-align:right;color:#6b7280'>Home &nbsp;&nbsp; Classification &nbsp;&nbsp; Model Info &nbsp;&nbsp; About</div>", unsafe_allow_html=True)

st.write("")
# Hero
left, right = st.columns([2,1])
with left:
    st.markdown("<small style='background:#fff0f6;padding:6px 10px;border-radius:8px;color:var(--muted)'>Deteksi Jenis</small>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-family:Inter, sans-serif;'>Deteksi Jenis <span style='color:var(--pink-500)'>Kendaraan AI</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280;max-width:640px'>Platform revolusioner yang menggunakan teknologi deep learning untuk mengidentifikasi dan mengklasifikasikan jenis kendaraan seperti mobil, motor, truck, dan bus dengan akurasi tinggi.</p>", unsafe_allow_html=True)
    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("<button class='btn-primary'>🚀 Coba Sekarang</button>", unsafe_allow_html=True)
    with c2:
        st.markdown("<button class='btn-ghost'>📘 Pelajari Lebih Lanjut</button>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'><h4 style='margin:0 0 8px 0'>Demo Cepat</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload gambar kendaraan untuk analisis", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar kendaraan diunggah", use_column_width=True)
        except Exception as e:
            st.error(f"Gagal membuka gambar: {e}")
    else:
        st.markdown("<div style='height:160px;display:flex;align-items:center;justify-content:center;color:#6b7280'>Upload gambar kendaraan untuk analisis</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("---")

# Cards (types)
st.subheader("Jenis Kendaraan yang Dapat Dideteksi")
st.markdown("<p style='color:#6b7280'>Sistem AI kami dapat mengenali berbagai jenis kendaraan dengan akurasi tinggi</p>", unsafe_allow_html=True)
colA, colB, colC, colD = st.columns(4)
colA.image("https://via.placeholder.com/200x100?text=Mobil")
colA.caption("Mobil – Sedan, SUV, Hatchback")
colB.image("https://via.placeholder.com/200x100?text=Motor")
colB.caption("Motor – Sepeda motor, skuter")
colC.image("https://via.placeholder.com/200x100?text=Truck")
colC.caption("Truck – Truk kargo, pickup")
colD.image("https://via.placeholder.com/200x100?text=Bus")
colD.caption("Bus – Bus kota, antar kota")

st.write("---")

# If user uploaded image and pressed a detect button, run inference
if uploaded_file is not None:
    st.subheader("Hasil Deteksi")
    col_img, col_info = st.columns([2,1])

    with col_img:
        # run inference
        start = time.time()
        pil_img = Image.open(uploaded_file).convert("RGB")
        boxes, scores, class_ids, class_names = run_yolo_inference(yolo_model, pil_img, imgsz=640, conf=0.25)
        yolo_time = (time.time() - start)

        # optional classifier on each crop
        classifier_labels, classifier_probs = run_classifier_on_crops(classifier, pil_img, boxes)

        # prepare labels to show: prefer classifier label if available, else yolo class name
        combined_labels = []
        for i in range(len(boxes)):
            lbl = None
            if classifier_labels and classifier_labels[i] is not None:
                p = classifier_probs[i]
                if p is not None:
                    lbl = f"{classifier_labels[i]} ({p:.2f})"
                else:
                    lbl = str(classifier_labels[i])
            else:
                lbl = class_names[i] if i < len(class_names) else (str(class_ids[i]) if i < len(class_ids) else "Object")
            combined_labels.append(lbl)

        # draw boxes
        img_with_boxes = pil_img.copy()
        img_with_boxes = draw_boxes_on_pil(img_with_boxes, boxes, labels=combined_labels, scores=scores)
        st.image(img_with_boxes, use_column_width=True)

    with col_info:
        st.markdown("**Ringkasan**")
        st.write(f"- Deteksi teridentifikasi: {len(boxes)}")
        st.write(f"- Waktu YOLO: {yolo_time*1000:.0f} ms (estimasi)")
        st.write(f"- Classifier aktif: {'Ya' if classifier is not None else 'Tidak'}")
        if len(boxes) > 0:
            st.write("---")
            st.markdown("**Detail Objek**")
            for i, box in enumerate(boxes):
                st.write(f"{i+1}. {combined_labels[i]} — Confidence YOLO: {scores[i]:.2f}")

st.write("---")
st.subheader("Mengapa Memilih Platform Kami?")
st.markdown("<p style='color:#6b7280'>Teknologi AI terdepan yang dirancang khusus untuk deteksi kendaraan dengan akurasi tinggi.</p>", unsafe_allow_html=True)
colx, coly, colz, cola = st.columns(4)
colx.metric("Akurasi Model", "98.2%")
coly.metric("Waktu Proses", "47ms")
colz.metric("Jenis Kendaraan", "4+")
cola.metric("Uptime", "99.9%")

st.write("")
st.markdown("© 2025 AI Image Detection — Built for demo", unsafe_allow_html=True)
