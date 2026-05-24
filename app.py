import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd      # [TAMBAHAN] Untuk membuat struktur data (DataFrame)
import altair as alt     # [TAMBAHAN] Untuk membuat visualisasi grafik (Dot Plot) interaktif

# Konfigurasi halaman utama Streamlit
st.set_page_config(
    page_title="Demo Klasifikasi Sampah 3R",
    page_icon="🎯",
    layout="centered"
)

# Kamus Pemetaan: 9 Kelas Model Latih -> 3 Kategori Utama Skripsi
PEMETAAN_KATEGORI = {
    'baterai': {'kategori': 'B3 (BERBAHAYA & BERACUN)', 'warna': '🔴', 'tips': 'Kumpulkan terpisah dan serahkan ke tempat pembuangan khusus limbah B3/DLH.'},
    'organik': {'kategori': 'ORGANIK', 'warna': '🟢', 'tips': 'Dapat diolah menjadi pupuk kompos atau pakan maggot.'},
    'kardus': {'kategori': 'ANORGANIK', 'warna': '🟡', 'tips': 'Setorkan ke Bank Sampah terdekat untuk didaur ulang.'},
    'baju': {'kategori': 'ANORGANIK', 'warna': '🟡', 'tips': 'Donasikan jika layak pakai atau kreasikan menjadi kain lap/kerajinan.'},
    'besi': {'kategori': 'ANORGANIK', 'warna': '🟡', 'tips': 'Setorkan ke pengepul rongsokan atau Bank Sampah untuk dilebur kembali.'},
    'kertas': {'kategori': 'ANORGANIK', 'warna': '🟡', 'tips': 'Hindari kondisi basah, kumpulkan untuk didaur ulang menjadi kertas baru.'},
    'plastik': {'kategori': 'ANORGANIK', 'warna': '🟡', 'tips': 'Batasi penggunaan (Reduce) atau bersihkan untuk wadah guna ulang (Reuse).'},
    'sepatu': {'kategori': 'ANORGANIK', 'warna': '🟡', 'tips': 'Perbaiki jika rusak ringan atau salurkan ke badan daur ulang tekstil.'},
    'kaca': {'kategori': 'ANORGANIK', 'warna': '🟡', 'tips': 'Hati-hati pecah, pisahkan agar aman saat diserahkan ke petugas kebersihan.'}
}

# Memuat model YOLO secara aman dengan sistem Cache Streamlit
@st.cache_resource
def load_model():
    try:
        return YOLO('best (2).onnx', task='classify')
    except Exception as e:
        st.error("❌ File model 'best.onnx' tidak ditemukan di direktori ini.")
        return None

model = load_model()

# Desain Antarmuka Dasbor
st.title("🎯 Media Pembelajaran 3R Interaktif")
st.write("Sistem Identifikasi Kategori Sampah Otomatis Menggunakan Algoritma YOLOv11s")
st.markdown("---")

# Menu Navigasi Pilihan Input Gambar
st.subheader("📸 Pilih Metode Masukan Gambar:")
opsi_input = st.radio(
    "Silakan pilih metode:",
    ("Ambil Foto Langsung (Kamera Device)", "Unggah Gambar dari Penyimpanan (Device Storage)"),
    label_visibility="collapsed"
)

image_data = None

# Logika Pemrosesan Kamera
if opsi_input == "Ambil Foto Langsung (Kamera Device)":
    camera_img = st.camera_input("Arahkan sampah ke kamera laptop/HP Anda:")
    if camera_img is not None:
        image_data = Image.open(camera_img).convert('RGB')

# Logika Pemrosesan Unggah File
else:
    uploaded_file = st.file_uploader("Pilih gambar sampah (Format: JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_data = Image.open(uploaded_file).convert('RGB')

# Proses Deteksi AI jika gambar tersedia
if image_data is not None:
    st.markdown("---")
    st.subheader("🖼️ Analisis Gambar Visual")
    
    if model is not None:
        with st.spinner("🔄 Algoritma YOLOv11s sedang menganalisis objek..."):
            # Konversi gambar PIL ke array numpy
            img_array = np.array(image_data)
            
            # Jalankan prediksi dengan YOLO (Klasifikasi)
            results = model.predict(source=img_array, verbose=False)
            
            # Ekstraksi hasil prediksi YOLO
            for r in results:
                label_asli = r.names[r.probs.top1].lower()
                tingkat_keyakinan = r.probs.top1conf.item() * 100
                
                # ==========================================================
                # [TRIK VISUAL] MENGGAMBAR BOUNDING BOX DENGAN OPENCV
                # ==========================================================
                # 1. Ubah format warna untuk OpenCV (RGB ke BGR)
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # 2. Ubah gambar ke Grayscale (Hitam Putih)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                
                # 3. Beri sedikit blur untuk mengurangi noise
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # 4. Gunakan Canny Edge Detection untuk mencari tepi objek
                edges = cv2.Canny(blur, 50, 150)
                
                # 5. Cari kontur (garis luar) dari objek
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 6. Jika kontur ditemukan, cari yang paling besar (diasumsikan itu sampahnya)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)
                    
                    # Jangan gambar kotak jika terlalu kecil (noise)
                    if w > 50 and h > 50:
                        # Gambar kotak hijau (Bounding Box)
                        cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        
                        # Gambar background hitam untuk teks label agar mudah dibaca
                        label_text = f"{label_asli.upper()} ({tingkat_keyakinan:.1f}%)"
                        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(img_cv, (x, y - text_h - 10), (x + text_w, y), (0, 255, 0), -1)
                        
                        # Tulis hasil YOLO di atas Bounding Box
                        cv2.putText(img_cv, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Kembalikan warna ke RGB untuk ditampilkan di Streamlit
                img_final = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                
                # Tampilkan gambar yang sudah ada Bounding Box-nya
                st.image(img_final, caption="Hasil Deteksi (OpenCV + YOLOv11s)", use_container_width=True)
                # ==========================================================

                # Mengambil pemetaan kategori utama skripsi
                info_sampah = PEMETAAN_KATEGORI.get(label_asli, {'kategori': 'TIDAK DIKETAHUI', 'warna': '⚪', 'tips': '-'})
                
                # Menampilkan Hasil Metrik
                st.success("✅ Analisis Selesai!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Nama Objek Terdeteksi", value=label_asli.upper())
                    st.metric(label="Tingkat Keyakinan AI", value=f"{tingkat_keyakinan:.2f}%")
                
                with col2:
                    st.metric(label="Kategori Sampah Skripsi", value=f"{info_sampah['warna']} {info_sampah['kategori']}")
                
                st.markdown("### 💡 Panduan Edukasi Pembelajaran 3R:")
                st.info(f"**Tindakan Pengelolaan:** {info_sampah['tips']}")
                
                # Bagian Dot Plot tetap dibiarkan seperti sebelumnya
                st.markdown("---")
                st.subheader("📊 Distribusi Keyakinan AI (Dot Plot)")
                st.caption("Grafik di bawah ini menunjukkan tingkat keyakinan (probabilitas) mesin terhadap seluruh 9 kelas objek yang dilatih.")
                
                semua_kelas = list(r.names.values())
                semua_probabilitas = r.probs.data.tolist()
                
                df_probs = pd.DataFrame({
                    'Kelas Sampah': [k.upper() for k in semua_kelas],
                    'Probabilitas (%)': [p * 100 for p in semua_probabilitas]
                }).sort_values(by='Probabilitas (%)', ascending=False)
                
                lines = alt.Chart(df_probs).mark_rule(color='gray', strokeDash=[3, 3]).encode(x='Probabilitas (%):Q', y=alt.Y('Kelas Sampah:N', sort='-x'))
                dots = alt.Chart(df_probs).mark_circle(size=200, opacity=1).encode(
                    x=alt.X('Probabilitas (%):Q', title='Tingkat Keyakinan (%)', scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y('Kelas Sampah:N', sort='-x', title='Objek'),
                    color=alt.Color('Kelas Sampah:N', legend=None),
                    tooltip=[alt.Tooltip('Kelas Sampah', title='Kelas'), alt.Tooltip('Probabilitas (%)', format='.2f')]
                )
                
                st.altair_chart(lines + dots, use_container_width=True)

    else:
        st.warning("⚠️ Proses klasifikasi dihentikan karena model AI belum termuat dengan benar.")

# Informasi Footer Akademik
st.markdown("---")
st.caption("Aplikasi Demo Skripsi © 2026 | Dikembangkan untuk Program Studi Teknik Informatika")
