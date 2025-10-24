import streamlit as st
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Konfigurasi halaman
# ==========================
st.set_page_config(
    page_title="AI Detection & Classification Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# Fungsi load model
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO(r"model/Adila Khairunnisa_Laporan 4.pt")  # YOLO untuk Apple & Tomato
    classifier = load_model(r"model/Adila Khairunnisa_Laporan 2.h5", compile=False, safe_mode=True)  # CNN penyakit kulit
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Tampilan utama
# ==========================
st.title("🧠 Smart Vision Dashboard")
st.markdown("### UAS Pemrograman Big Data — Adila Khairunnisa 🌟")

with st.expander("📁 Dataset yang Digunakan"):
    st.markdown("""
    **1️⃣ YOLO (.pt model)** — *Waste Classification Dataset*  
    - Kelas: **Apple**, **Tomato**  

    **2️⃣ CNN (.h5 model)** — *Skin Disease Dataset*  
    - Kelas: **Eczema**, **Acne**, **Milia**, **Rosacea**, **Keratosis**, **Carcinoma**
    """)

st.divider()

# ==========================
# Sidebar Menu
# ==========================
menu = st.sidebar.radio("🔍 Pilih Mode Analisis", ["Deteksi Objek (YOLO)", "Klasifikasi Penyakit Kulit"])
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# Proses Gambar
# ==========================
if uploaded_file is not None:
    col1, col2 = st.columns(2)
    img = Image.open(uploaded_file).convert("RGB")
    col1.image(img, caption="Gambar Asli", use_container_width=True)

    # ==============================
    # MODE YOLO (Apple - Tomato)
    # ==============================
    if menu == "Deteksi Objek (YOLO)":
        st.subheader("🍎 Deteksi Apple dan Tomato")

        with st.spinner("🔍 Mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()
            detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []

        # Kalau YOLO tidak yakin atau tidak mendeteksi objek valid
        if len(detections) == 0 or np.max(results[0].boxes.conf.cpu().numpy()) < 0.6:
            st.warning("⚠️ Tidak ada objek Apple/Tomato yang terdeteksi dengan yakin. Beralih ke klasifikasi penyakit kulit...")
            
            # Jalankan mode klasifikasi otomatis
            class_labels = ["Eczema", "Acne", "Milia", "Rosacea", "Keratosis", "Carcinoma"]
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            st.success("✅ Klasifikasi selesai!")
            st.markdown(f"### Hasil Prediksi: **{class_labels[class_index]}**")
            st.progress(float(confidence))
            st.caption(f"Akurasi: {confidence:.2%}")

        else:
            col2.image(result_img, caption="Hasil Deteksi", use_container_width=True)
            st.success("✅ Deteksi selesai!")

            # tampilkan kelas yang terdeteksi
            detected_classes = list(set([yolo_model.names[int(box.cls)] for box in results[0].boxes]))
            st.write("**Kelas yang terdeteksi:**", ", ".join(detected_classes))

    # ==============================
    # MODE CNN Klasifikasi Kulit
    # ==============================
    elif menu == "Klasifikasi Penyakit Kulit":
        st.subheader("🩺 Klasifikasi Penyakit Kulit")

        class_labels = ["Eczema", "Acne", "Milia", "Rosacea", "Keratosis", "Carcinoma"]

        with st.spinner("🧠 Menganalisis gambar..."):
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

        st.success("✅ Prediksi selesai!")
        st.markdown(f"### Hasil Prediksi: **{class_labels[class_index]}**")
        st.progress(float(confidence))
        st.caption(f"Akurasi: {confidence:.2%}")

        # tampilkan grafik probabilitas
        st.markdown("#### Distribusi Probabilitas:")
        probs = {class_labels[i]: float(prediction[0][i]) for i in range(len(class_labels))}
        st.bar_chart(probs)

else:
    st.warning("⚠️ Silakan unggah gambar terlebih dahulu untuk mulai analisis.")
