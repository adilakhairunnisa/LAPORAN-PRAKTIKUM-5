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
st.write("Aplikasi ini menggabungkan **deteksi objek (YOLO)** dan **klasifikasi gambar (CNN)** untuk dua dataset berbeda:")

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
st.sidebar.info("Unggah gambar di bawah untuk memulai analisis!")

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# Proses Gambar
# ==========================
if uploaded_file is not None:
    col1, col2 = st.columns(2)
    img = Image.open(uploaded_file).convert("RGB")
    col1.image(img, caption="Gambar Asli", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        st.subheader("🍎 Deteksi Apple dan Tomato")

        with st.spinner("🔍 Mendeteksi objek..."):
            results = yolo_model(img)
            result_img = results[0].plot()

        col2.image(result_img, caption="Hasil Deteksi", use_container_width=True)

        st.success("✅ Deteksi selesai!")
        st.write("**Kelas yang terdeteksi:**")
        detected_classes = list(set([r.names[int(box.cls)] for r in results for box in r.boxes]))
        st.write(", ".join(detected_classes) if detected_classes else "_Tidak ada objek terdeteksi_")

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

        # Tampilkan probabilitas semua kelas
        st.markdown("#### Distribusi Probabilitas:")
        probs = {class_labels[i]: float(prediction[0][i]) for i in range(len(class_labels))}
        st.bar_chart(probs)

else:
    st.warning("⚠️ Silakan unggah gambar terlebih dahulu untuk mulai analisis.")
