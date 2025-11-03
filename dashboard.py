import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["DISPLAY"] = ":0"
os.system("apt-get update -y && apt-get install -y libgl1-mesa-glx libglib2.0-0 > /dev/null 2>&1 || true")

import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Adila Khairunnisa_Laporan 4.pt")  # Model YOLO
    classifier = tf.keras.models.load_model("model/Adila Khairunnisa_Laporan 2.h5", compile=False)  # Model CNN
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.set_page_config(page_title="🧠 Image Analyzer", page_icon="🧠", layout="wide")
st.title("🧠 Dashboard Deteksi & Klasifikasi Gambar")

menu = st.sidebar.selectbox("📊 Pilih Mode:", [
    "Deteksi Buah (Apple & Tomato)",
    "Klasifikasi Penyakit Kulit"
])

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    # ==========================
    # MODE 1: Deteksi Buah (YOLO)
    # ==========================
    if menu == "Deteksi Buah (Apple & Tomato)":
        st.subheader("🍎 Deteksi Apple & Tomato")
        with st.spinner("Mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()
        st.image(result_img, caption="📍 Hasil Deteksi", use_container_width=True)

        detected_classes = list(set([yolo_model.names[int(box.cls)] for box in results[0].boxes]))
        if detected_classes:
            st.success("✅ Deteksi selesai!")
            st.write("**Kelas terdeteksi:**", ", ".join(detected_classes))
        else:
            st.warning("⚠️ Tidak ada objek terdeteksi.")

    # ==========================
    # MODE 2: Klasifikasi Penyakit Kulit
    # ==========================
    elif menu == "Klasifikasi Penyakit Kulit":
        st.subheader("🩺 Klasifikasi Jenis Penyakit Kulit")

        # --- SESUAIKAN UKURAN INI DENGAN MODEL KAMU ---
        # Kalau model.input_shape = (None, 128, 128, 3), ubah jadi (128, 128)
        # Kalau model.input_shape = (None, 224, 224, 3), biarkan (224, 224)
        img_resized = img.resize((224, 224))
        # ----------------------------------------------

        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        with st.spinner("🧠 Mengklasifikasi gambar..."):
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

        labels = ["Eczema", "Acne", "Milia", "Rosacea", "Keratosis", "Carcinoma"]
        predicted_label = labels[class_index]

        st.success("✅ Klasifikasi selesai!")
        st.markdown(f"### 🧩 Hasil: **{predicted_label}**")
        st.write(f"**Probabilitas:** {confidence:.2%}")

        st.markdown("#### 📊 Distribusi Prediksi:")
        probs = {labels[i]: float(prediction[0][i]) for i in range(len(labels))}
        st.bar_chart(probs)

else:
    st.info("💡 Silakan unggah gambar terlebih dahulu untuk memulai analisis.")
