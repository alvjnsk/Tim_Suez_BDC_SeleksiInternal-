# MBG Tweet Classifier — Satria Data BDC 2026

## Identitas Tim
- **Nama Tim**   : Tim Suez
- **Anggota**    : [ISI NAMA ANGGOTA]
- **Institusi**  : Telkom University
- **Divisi**     : Big Data Challenge (BDC) — Case 1

---

## Deskripsi
Solusi klasifikasi tweet tentang program Makan Bergizi Gratis (MBG) ke dalam 8 kelas menggunakan fine-tuning IndoBERT dengan Weighted CrossEntropyLoss untuk menangani class imbalance. Model dibangun menggunakan pendekatan transfer learning berbasis pretrained language model IndoBERT yang telah terbukti efektif untuk teks Bahasa Indonesia informal.

**8 Kelas Target:**
1. Anggaran
2. Kualitas Pangan
3. Distribusi
4. Ekonomi
5. Tata Kelola
6. Sasaran Penerima
7. Politik
8. Lainnya

---

## Requirements

### Library yang Dibutuhkan
```
torch
transformers==4.41.0
datasets
accelerate
openpyxl
scikit-learn
pandas
seaborn
matplotlib
```

### Install Semua Library
```bash
pip install transformers==4.41.0 datasets accelerate openpyxl scikit-learn pandas seaborn matplotlib
```

---

## Struktur File
```
├── Tim_Suez_MBG_Classifier_Colab.ipynb  # Notebook utama
├── case_1_labeled_data.xlsx        # Data training (5.000 tweet berlabel)
├── case_1_text_to_predict.xlsx     # Data test (1.500 tweet tanpa label)
├── case_1_template_sheet.xlsx      # Template submission panitia
├── Tim_suez.xlsx                   # File prediksi hasil (output)
└── README.md                       # Dokumentasi ini
```

---

## Cara Menjalankan

### Di Google Colab (Rekomendasi)
1. Buka [https://colab.research.google.com](https://colab.research.google.com)
2. Klik `File → Upload notebook` → pilih `Tim_Suez_MBG_Classifier_Colab.ipynb`
3. Aktifkan GPU: `Runtime → Change runtime type → T4 GPU → Save`
4. Jalankan **Cell 1** untuk install library
5. Jalankan **Cell 2** untuk import library
6. Jalankan **Cell 3** → upload ketiga file dataset saat diminta
7. Jalankan semua cell secara berurutan dari Cell 4 sampai Cell 13
8. File submission otomatis terdownload setelah Cell 13 selesai

### Catatan Penting
- GPU **wajib** diaktifkan — tanpa GPU training bisa memakan waktu 8–15 jam
- Jangan tutup tab browser saat proses training berlangsung
- Estimasi waktu training: **30–60 menit** dengan GPU T4
- Pastikan ketiga file dataset tersedia sebelum menjalankan Cell 3


## Metodologi

### 1. Pre-processing

| Tahap | Teknik | Detail |
|-------|--------|--------|
| Cleaning | Hapus URL | `http://...`, `https://...`, `www...` |
| Cleaning | Hapus mention | `@username` |
| Cleaning | Hapus hashtag simbol | `#MBG` → `MBG` |
| Cleaning | Hapus emoji & non-ASCII | Karakter di luar ASCII |
| Cleaning | Hapus karakter berulang | `bagussss` → `baguss` |
| Cleaning | Hapus tanda baca berlebihan | Simbol selain `. , ! ?` |
| Normalisasi | Lowercase | `GAGAL` → `gagal` |
| Normalisasi | Normalisasi spasi | Spasi ganda → spasi tunggal |
| Transformasi | Tokenisasi | Via IndoBERT tokenizer |
| Transformasi | Vektorisasi | Via IndoBERT embedding (768 dimensi) |

### 2. Model

- **Base Model** : IndoBERT (`indobenchmark/indobert-base-p1`)
- **Task**       : Multiclass Sequence Classification (8 kelas)
- **Metode**     : Fine-tuning supervised pada 5.000 tweet berlabel
- **Output**     : Probabilitas per kelas → argmax → label prediksi

### 3. Penanganan Class Imbalance

Dataset memiliki distribusi kelas yang tidak seimbang:

| Kelas | Jumlah Tweet |
|-------|-------------|
| Kualitas Pangan | 1.247 |
| Politik | 792 |
| Anggaran | 727 |
| Lainnya | 638 |
| Tata Kelola | 511 |
| Sasaran Penerima | 507 |
| Distribusi | 433 |
| Ekonomi | 145 |

**Solusi:** Weighted CrossEntropyLoss — kelas dengan data sedikit diberi bobot lebih tinggi sehingga model lebih memperhatikan kelas minoritas.

```
Weight kelas = Total data / (Jumlah kelas × Jumlah data kelas)
```

### 4. Konfigurasi Training Terbaik

| Parameter | Nilai |
|-----------|-------|
| Model | indobenchmark/indobert-base-p1 |
| Epochs | 5 |
| Learning Rate | 2e-5 |
| Batch Size | 32 |
| Max Token Length | 128 |
| Optimizer | AdamW (weight decay 0.01) |
| Scheduler | Linear warmup (10%) + linear decay |
| Validation Split | 10% (stratified) |
| Loss Function | Weighted CrossEntropyLoss |
| Best Model | Disimpan berdasarkan Val Balanced Accuracy tertinggi |

---

## Hasil Eksperimen

Berikut seluruh eksperimen yang dilakukan beserta hasilnya:

| # | Konfigurasi | Balanced Accuracy |
|---|-------------|:-----------------:|
| 1 | IndoBERT Base + CrossEntropyLoss | **71.08%**  Terbaik |
| 2 | IndoBERT Base + CrossEntropyLoss (rerun) | 70.07% |
| 3 | IndoBERT Base + Normalisasi Slang | 65.75% |
| 4 | XLM-RoBERTa, Epoch 5 | 65.92% |
| 5 | XLM-RoBERTa, Epoch 10, LR 1e-5 | 66.17% |
| 6 | IndoBERT Base + Stemming & Stopword Removal | 63.41% |
| 7 | IndoBERT Base, Batch 16, Epoch 7 | 67.95% |
| 8 | IndoBERT Base + Focal Loss + Random Seed | 69.10% |
| 9 | IndoBERT Large | 69.67% |

### Detail Hasil Terbaik (71.08%)

```
Balanced Accuracy: 0.7108 (71.08%)

                  precision    recall  f1-score   support
        Anggaran      0.836     0.836     0.836        73
      Distribusi      0.607     0.860     0.712        43
         Ekonomi      0.565     0.929     0.703        14
 Kualitas Pangan      0.864     0.608     0.714       125
         Lainnya      0.684     0.609     0.645        64
         Politik      0.584     0.570     0.577        79
Sasaran Penerima      0.544     0.608     0.574        51
     Tata Kelola      0.531     0.667     0.591        51
```

---

## Kesimpulan Eksperimen

1. **Preprocessing minimal lebih baik untuk BERT** — Stemming dan stopword removal justru menurunkan performa karena menghilangkan konteks yang dibutuhkan model BERT untuk memahami makna kalimat.

2. **IndoBERT Base lebih baik dari Large untuk dataset kecil** — IndoBERT Large memiliki lebih banyak parameter tapi membutuhkan lebih banyak data untuk konvergen optimal.

3. **Weighted CrossEntropyLoss efektif menangani imbalance** — Terbukti dari recall kelas Ekonomi (data paling sedikit) yang tetap tinggi di 0.929.

4. **Kelas Politik dan Lainnya paling sulit** — Kedua kelas ini memiliki recall terendah karena topiknya ambigu dan sering overlap dengan kelas lain.

---

## Referensi
- **IndoBERT**: Wilie et al. (2020) — IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding
  [https://huggingface.co/indobenchmark/indobert-base-p1](https://huggingface.co/indobenchmark/indobert-base-p1)
- **HuggingFace Transformers**: Wolf et al. (2020) — Transformers: State-of-the-Art Natural Language Processing
  [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- **Focal Loss**: Lin et al. (2017) — Focal Loss for Dense Object Detection
- **AdamW Optimizer**: Loshchilov & Hutter (2019) — Decoupled Weight Decay Regularization
