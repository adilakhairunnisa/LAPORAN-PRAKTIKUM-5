# ==============================
# PREVENT OPENCV ERROR (WAJIB UNTUK STREAMLIT CLOUD)
# ==============================
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["DISPLAY"] = ":0"

# ==============================
# IMPORT LIBRARY
# ==============================
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==============================
# LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Adila Khairunnisa_Laporan 4.pt")  # Model YOLO
    classifier = tf.keras.models.load_model("model/Adila Khairunnisa_Laporan 2.h5", compile=False)  # Model CNN
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==============================
# UI SETUP
# ==============================
st.set_page_config(page_title="🧠 Image Analyzer", page_icon="🧠", layout="wide")
st.title("🧠 Dashboard Deteksi & Klasifikasi Gambar")

menu = st.sidebar.selectbox(
    "📊 Pilih Mode Analisis:",
    ["Deteksi Buah (Apple & Tomato)", "Klasifikasi Penyakit Kulit"]
)

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==============================
# LOGIKA UTAMA
# ==============================
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    # === MODE 1: DETEKSI BUAH ===
    if menu == "Deteksi Buah (Apple & Tomato)":
        st.subheader("🍎 Deteksi Apple & Tomato")
        with st.spinner("🔍 Mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()

        st.image(result_img, caption="📍 Hasil Deteksi", use_container_width=True)

        detected_classes = list(set([yolo_model.names[int(box.cls)] for box in results[0].boxes]))
        if detected_classes:
            st.success("✅ Deteksi selesai!")
            st.write("**Kelas terdeteksi:**", ", ".join(detected_classes))
        else:
            st.warning("⚠️ Tidak ada objek terdeteksi.")

    # === MODE 2: KLASIFIKASI PENYAKIT KULIT ===
    elif menu == "Klasifikasi Penyakit Kulit":
        st.subheader("🩺 Klasifikasi Jenis Penyakit Kulit")

        img_resized = img.resize((224, 224))
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

        probs = {labels[i]: float(prediction[0][i]) for i in range(len(labels))}
        st.bar_chart(probs)

else:
    st.info("💡 Silakan unggah gambar terlebih dahulu untuk memulai analisis.")
