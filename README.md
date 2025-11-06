# UTS Kecerdasan Buatan

📌 Deskripsi Proyek

Proyek ini bertujuan untuk mendeteksi ujaran kebencian (hate speech) dalam teks media sosial menggunakan teknik Natural Language Processing (NLP) dan Machine Learning.
Model dikembangkan untuk mengklasifikasikan teks ke dalam dua kategori:

- 0 → Tidak mengandung ujaran kebencian
- 1 → Mengandung ujaran kebencian

Metode yang digunakan menggabungkan representasi fitur TF-IDF dan algoritma klasifikasi Logistic Regression untuk mencapai hasil yang akurat, efisien, dan mudah diimplementasikan.

⚙️ Fitur Utama
- Pembersihan teks otomatis (case folding, hapus tanda baca, angka, URL, stopwords)
- Representasi fitur menggunakan TF-IDF Vectorizer
- Model klasifikasi menggunakan Logistic Regression
- Evaluasi performa dengan metrik Accuracy, Precision, Recall, dan F1-Score
- Visualisasi hasil menggunakan Confusion Matrix dan Word Cloud

🧩 Langkah Implementasi

1. Dataset
Menggunakan dataset publik seperti Hate Speech and Offensive Language Dataset (Davidson et al.) atau dataset lokal berbahasa Indonesia.
Label yang digunakan:

- 0 = Tidak mengandung ujaran kebencian
- 1 = Mengandung ujaran kebencian

2. Preprocessing Teks
- Case Folding
- Hapus tanda baca, angka, URL, mention, dan hashtag
- Stopword Removal
- Tokenisasi

3. Representasi Fitur
Menggunakan TF-IDF (Term Frequency–Inverse Document Frequency) untuk mengubah teks menjadi vektor numerik.

4. Pembangunan Model
Menggunakan Logistic Regression dengan data latih 80% dan data uji 20%.

5. Evaluasi Model
Menggunakan metrik:
- Accuracy
- Precision
- Recall
- F1-Score

6. Visualisasi
- Distribusi label dataset (diagram pie)
- Word Cloud untuk kata yang sering muncul pada ujaran kebencian
- Confusion Matrix hasil prediksi
  
