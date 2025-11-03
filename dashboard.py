import streamlit as st
from streamlit_option_menu import option_menu
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_path = "model/Adila_Khairunnisa_Laporan4.pt"  # model deteksi apel & tomat
    h5_path = "model/Adila_Khairunnisa_Laporan2.h5"   # model klasifikasi penyakit kulit
    yolo_model = YOLO(yolo_path)
    classifier = tf.keras.models.load_model(h5_path)
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Sidebar Navigation
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "Home"

with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Classification", "Detection"],
        icons=["house", "stethoscope", "apple"],
        menu_icon="cast",
        default_index=0
    )
    st.session_state.page = selected

# ==========================
# Home Page
# ==========================
if st.session_state.page == "Home":
    st.image(
        "https://upload.wikimedia.org/wikipedia/id/2/2c/Logo_Universitas_Syiah_Kuala.png",
        width=150,
    )
    st.title("Universitas Syiah Kuala")
    st.subheader("Praktikum Big Data")

    st.write("**Nama:** Adila Khairunnisa")
    st.write("**NPM:** 2208108010010")

    st.markdown("""
    Selamat datang 👋  
    - 🧴 **Skin Disease Classifier (Klasifikasi Jenis Penyakit Kulit)**  
    - 🍎 **Fruit Detector (Deteksi Apel dan Tomat)**  
    """)

    st.markdown("""
    ---
    ### 🧭 Panduan Penggunaan:
    1. Pilih menu di **sidebar kiri**:
        - **Classification** → untuk mengenali jenis penyakit kulit.  
        - **Detection** → untuk mendeteksi objek buah (apel/tomat).  
    2. Unggah gambar (JPG/PNG).  
    3. Tunggu hasil prediksi.  
    4. Lihat hasil dan confidence-nya.  
    5. Klik **Kembali ke Beranda** untuk fitur lain.  
    ---
    """)

    st.info("💡 Tips: Gunakan gambar dengan pencahayaan dan kualitas baik untuk hasil akurat.")

# ==========================
# Skin Disease Classification
# ==========================
elif st.session_state.page == "Classification":
    st.markdown("<h2 style='text-align:center;'>🧴 Klasifikasi Jenis Penyakit Kulit</h2>", unsafe_allow_html=True)
    st.write("Unggah gambar kulit untuk mengenali jenis penyakit kulit menggunakan model klasifikasi.")

    uploaded_file = st.file_uploader("Unggah gambar kulit", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)
        st.write("🔍 Sedang memproses...")

        time.sleep(1)

        # Sesuaikan input dengan model H5
        target_size = classifier.input_shape[1:3]
        img_resized = img.resize(target_size)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediksi penyakit
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        # Misalnya urutan kelas model kamu begini:
        classes = ['Normal', 'Acne', 'Eczema', 'Psoriasis', 'Melanoma']

        st.progress(int(confidence * 100))
        st.write(f"**Confidence:** {confidence:.2%}")

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #e3f2fd, #fce4ec); 
                    padding:25px; border-radius:20px; text-align:center;'>
        <h3>🩺 Hasil Klasifikasi:</h3>
        <h2 style='color:#2e7d32'>{classes[class_index]}</h2>
        <p>Prediksi model menunjukkan kemungkinan kondisi kulit seperti di atas.</p>
        </div>
        """, unsafe_allow_html=True)

        st.balloons()
        st.success("✨ Analisis selesai!")

    if st.button("🏠 Kembali ke Beranda"):
        st.session_state.page = "Home"

# ==========================
# Fruit Detection
# ==========================
elif st.session_state.page == "Detection":
    st.markdown("<h2 style='text-align:center;'>🍎 Deteksi Apel dan Tomat</h2>", unsafe_allow_html=True)
    st.write("Unggah gambar buah untuk mendeteksi apakah itu **apel** atau **tomat**.")

    option = st.radio("Pilih sumber gambar:", ["Unggah dari File", "Ambil dari Kamera"])

    if option == "Unggah dari File":
        uploaded_fruit = st.file_uploader("Unggah gambar buah", type=["jpg", "jpeg", "png"])
        if uploaded_fruit:
            img_fruit = Image.open(uploaded_fruit)
        else:
            img_fruit = None
    else:
        camera_photo = st.camera_input("Ambil foto buah menggunakan kamera")
        if camera_photo:
            img_fruit = Image.open(camera_photo)
        else:
            img_fruit = None

    if img_fruit:
        st.image(img_fruit, caption="Gambar yang digunakan", use_container_width=True)
        st.write("🔎 Sedang mendeteksi...")
        time.sleep(1)

        results = yolo_model(np.array(img_fruit))
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

        detected_label = "Tidak terdeteksi"
        if len(results[0].boxes) > 0:
            cls_id = int(results[0].boxes.cls[0])
            label_names = ["Apple", "Tomato"]  # urutan sesuai model kamu
            detected_label = label_names[cls_id]

        if detected_label == "Apple":
            st.markdown("""
                <div style='background: linear-gradient(135deg, #f1f8e9, #c5e1a5); padding:25px; border-radius:20px; text-align:center'>
                🍏 <h3>Buah yang terdeteksi: <b>APEL</b></h3>
                <p>Apel mengandung serat dan vitamin C tinggi. Segar dan sehat!</p>
                </div>
            """, unsafe_allow_html=True)
        elif detected_label == "Tomato":
            st.markdown("""
                <div style='background: linear-gradient(135deg, #ffebee, #ef9a9a); padding:25px; border-radius:20px; text-align:center'>
                🍅 <h3>Buah yang terdeteksi: <b>TOMAT</b></h3>
                <p>Tomat kaya akan likopen, baik untuk kulit dan kesehatan jantung!</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Tidak terdeteksi buah dengan jelas. Coba ulangi dengan gambar lain.")

        st.success("✅ Deteksi selesai!")

    if st.button("🏠 Kembali ke Beranda"):
        st.session_state.page = "Home"
