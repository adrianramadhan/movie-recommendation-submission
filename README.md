# Laporan Proyek Machine Learning – Adrian Putra Ramadhan

## Project Overview
Di era platform streaming dan big‑budget productions, investasi untuk membuat satu film bisa mencapai ratusan juta dolar. Meski begitu, banyak judul blockbuster tetap gagal secara komersial (“box‑office flop”), menimbulkan kerugian finansial besar. Di sisi lain, penonton menghadapi ribuan pilihan judul yang membuat proses memilih film menjadi sulit (paradox of choice). Sistem rekomendasi film membantu:  
1. **Pengguna** menemukan judul yang sesuai minatnya.  
2. **Studio/platform** meminimalkan risiko flop dengan memprediksi tema dan genre yang berpotensi laku.  

**Referensi Terkait**:  
- [The Netflix Recommender System: Algorithms, Business Value, and Innovation](https://ailab-ua.github.io/courses/resources/netflix_recommender_system_tmis_2015.pdf)  
- [A Survey of Movie Recommendation Techniques](https://medium.com/@akshaymouryaart/a-survey-on-movie-recommendation-system-d9610777f8e5)  

## Business Understanding
### Problem Statements
1. Bagaimana merekomendasikan film yang relevan kepada pengguna hanya berdasarkan metadata film (*tanpa data interaksi*)?  
2. Bagaimana memanfaatkan atribut genre, keywords, overview, dan popularity untuk menghasilkan daftar **Top‑N rekomendasi**?  

### Goals
1. Membangun pipeline ekstraksi fitur dari metadata:  
   - Genres: one‑hot encoding multi‑label  
   - Keywords & Overview: vektorisasi TF‑IDF  
   - Popularity: skala numerik langsung  
2. Mengimplementasikan **Content‑based Filtering** menggunakan cosine similarity untuk menghasilkan Top‑N film mirip.  

### Solution Approach
#### Feature Extraction
- **Genres**: One‑hot encoding multi‑label.  
- **Keywords & Overview**: Vektorisasi TF‑IDF.  
- **Popularity**: Skala numerik langsung.  

#### Similarity Calculation
1. Gabungkan semua vektor fitur → matriks `film × fitur`.  
2. Hitung **cosine similarity** antar baris (film).  

#### Top‑N Recommendation
- Untuk setiap film input sederhana (list film favorit), pilih film dengan similarity tertinggi.


## Data Understanding
### Sumber Data
Dataset: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)  
Jumlah Data: 4803 film

### Daftar Lengkap Variabel/Fitur
Berikut adalah metadata lengkap yang tersedia dalam dataset:

| No | Kolom | Tipe Data | Non-Null Count | Deskripsi | Contoh Data |
|----|-------|-----------|----------------|-----------|-------------|
| 1 | budget | int64 | 4803 | Anggaran produksi film (dalam USD) | 237000000 |
| 2 | genres | object | 4803 | Daftar genre dalam format JSON | `[{"id":28,"name":"Action"},{"id":12,"name":"Adventure"}]` |
| 3 | homepage | object | 1712 | URL website resmi film | "http://www.avatarmovie.com/" |
| 4 | id | int64 | 4803 | ID unik film di TMDB | 19995 |
| 5 | keywords | object | 4803 | Kata kunci terkait film (JSON) | `[{"id":1463,"name":"culture clash"},{"id":2968,"name":"future"}]` |
| 6 | original_language | object | 4803 | Bahasa asli film (kode ISO) | "en" |
| 7 | original_title | object | 4803 | Judul asli film | "Avatar" |
| 8 | overview | object | 4800 | Sinopsis/ringkasan cerita | "In the 22nd century, a paraplegic Marine..." |
| 9 | popularity | float64 | 4803 | Skor popularitas TMDB | 150.437577 |
| 10 | production_companies | object | 4803 | Perusahaan produksi (JSON) | `[{"name":"Ingenious Film Partners","id":289}]` |
| 11 | production_countries | object | 4803 | Negara produksi (JSON) | `[{"iso_3166_1":"US","name":"United States"}]` |
| 12 | release_date | object | 4802 | Tanggal rilis (YYYY-MM-DD) | "2009-12-10" |
| 13 | revenue | int64 | 4803 | Pendapatan kotor (USD) | 2787965087 |
| 14 | runtime | float64 | 4801 | Durasi film (menit) | 162.0 |
| 15 | spoken_languages | object | 4803 | Bahasa yang digunakan (JSON) | `[{"iso_639_1":"en","name":"English"}]` |
| 16 | status | object | 4803 | Status rilis | "Released" |
| 17 | tagline | object | 3959 | Slogan film | "Enter the World of Pandora." |
| 18 | title | object | 4803 | Judul film | "Avatar" |
| 19 | vote_average | float64 | 4803 | Rating rata-rata (0-10) | 7.2 |
| 20 | vote_count | int64 | 4803 | Jumlah vote | 11800 |

### Kualitas Data:
   - Tidak ada data duplikat
   - Missing values:

| Kolom | Jumlah Missing |
|-------|----------------|
| homepage | 3091 |
| tagline | 844 |
| overview | 3 |
| release_date | 1 |
| runtime | 2 |

### Eksplorasi Data
**Distribusi Genre**:
   - Genre paling umum: Drama, Comedy, Thriller
   - Visualisasi distribusi 10 besar genre:

   ![Top 10 Genres](./image/image.png)


## Data Preparation

---

### 1. Ekstraksi dan Transformasi Data

#### Proses yang Dilakukan:
- Membuat **salinan dataset asli** untuk menjaga keutuhan data sumber.
- Mengekstrak data JSON dari kolom `keywords` menjadi daftar kata kunci yang terstruktur.
- Menangani nilai kosong pada kolom `overview` dengan **string kosong**.
- Memilih hanya **fitur relevan** untuk sistem rekomendasi:
  - `title`: Nama film (sebagai identifikasi)
  - `genres`: Daftar genre
  - `keywords`: Daftar kata kunci
  - `overview`: Sinopsis film
  - `popularity`: Skor numerik popularitas

#### Alasan:
- Salinan data mencegah perubahan tidak disengaja.
- Ekstraksi JSON mengubah struktur nested menjadi format siap olah.
- Null handling mencegah error saat proses teks.
- Seleksi fitur memfokuskan pada informasi semantik yang paling berguna.

---

### 2. Feature Engineering

#### 2.1 Pengkodean Genre
- Mengubah daftar genre menjadi **matriks biner multi-label**:
  - Genre → Kolom fitur
  - Nilai 1 jika film memiliki genre tersebut, 0 jika tidak

#### 2.2 Vektorisasi Teks
#### Ringkasan Film (`overview`)
- Menggunakan **TF-IDF Vectorization**
- Menghapus kata umum (stop words)
- Membatasi maksimal **5000 fitur**

#### Kata Kunci (`keywords`)
- Menggabungkan daftar kata kunci menjadi string
- Menggunakan TF-IDF terpisah
- Membatasi maksimal **3000 fitur**

#### 2.3 Penggabungan Fitur
Menggabungkan:
- Matriks biner genre
- TF-IDF ringkasan
- TF-IDF kata kunci
- **Skor popularitas** (dinormalisasi)

Hasil: Matriks fitur akhir yang merepresentasikan film secara semantik dan numerik

---

### 3. Hasil Akhir Data Preparation

---

#### Alasan Teknis

| Langkah | Tujuan |
|--------|--------|
| Pembuatan Salinan | Mencegah modifikasi tidak disengaja |
| Ekstraksi JSON | Mengurai struktur kompleks jadi format analisis |
| Pembersihan Teks | Menjamin konsistensi dan mencegah error |
| Seleksi Fitur | Fokus pada atribut paling informatif |
| Pengkodean Genre | Representasi multi-label yang sesuai |
| TF-IDF | Tangkap bobot semantik kata penting |
| Batasan Fitur | Cegah overfitting dan kurangi noise |
| Integrasi Fitur | Representasi lengkap untuk perhitungan similarity |

---

#### 🚀 Kesimpulan

Pipeline ini mengubah metadata film mentah menjadi **matriks fitur vektor** yang optimal untuk digunakan dalam algoritma **Content-Based Filtering**, menjaga konteks semantik antar film, dan siap digunakan untuk perhitungan kemiripan antar film.
