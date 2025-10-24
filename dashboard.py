import streamlit as st
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    # Model YOLO untuk deteksi penyakit kulit
    yolo_model = YOLO("model/Adila Khairunnisa_Laporan 4.pt")

    # Model klasifikasi Apple vs Tomato
    classifier = load_model("model/Adila Khairunnisa_Laporan 2.h5", compile=False, safe_mode=True)
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
st.set_page_config(page_title="AI Detection Dashboard", layout="wide")
st.title("🧠 AI Detection & Classification Dashboard")

menu = st.sidebar.radio("🔍 Pilih Mode Analisis", ["Deteksi Penyakit Kulit (YOLO)", "Klasifikasi Buah (Apple vs Tomato)"])

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Penyakit Kulit (YOLO)":
        st.subheader("🧴 Deteksi Penyakit Kulit")
        st.info("Model YOLO digunakan untuk mendeteksi enam jenis penyakit kulit: Eczema, Acne, Milia, Rosacea, Keratosis, dan Carcinoma.")

        results = yolo_model(img)
        result_img = results[0].plot()
        st.image(result_img, caption="🩺 Hasil Deteksi Penyakit Kulit", use_container_width=True)

        # Tampilkan label hasil deteksi
        detected_classes = list(set([yolo_model.names[int(box.cls)] for box in results[0].boxes]))
        if detected_classes:
            st.success("✅ Deteksi selesai!")
            st.write("**Kelas terdeteksi:**", ", ".join(detected_classes))
        else:
            st.warning("⚠️ Tidak ada penyakit kulit yang terdeteksi.")

    elif menu == "Klasifikasi Buah (Apple vs Tomato)":
        st.subheader("🍎 Klasifikasi Buah (Apple vs Tomato)")
        st.info("Model CNN digunakan untuk mengklasifikasikan gambar ke dalam dua kelas: Apple dan Tomato.")

        # Preprocessing
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_names = ["Apple", "Tomato"]
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success("✅ Klasifikasi selesai!")
        st.metric(label="Kelas Prediksi", value=class_names[class_index])
        st.progress(float(confidence))
        st.write("**Probabilitas:**", f"{confidence*100:.2f}%")

else:
    st.warning("📁 Silakan unggah gambar terlebih dahulu untuk mulai analisis.")
