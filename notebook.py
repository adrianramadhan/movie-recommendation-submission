#!/usr/bin/env python
# coding: utf-8

# ## Sistem Rekomendasi TMDB Movie
# 
# Di era platform streaming dan big‑budget productions, investasi untuk membuat satu film bisa mencapai ratusan juta dolar. Meski begitu, banyak judul blockbuster tetap gagal secara komersial (“box‑office flop”), menimbulkan kerugian finansial besar. Di sisi lain, penonton menghadapi ribuan pilihan judul yang membuat proses memilih film menjadi sulit (paradox of choice). Sistem rekomendasi film membantu:  
# 1. **Pengguna** menemukan judul yang sesuai minatnya.  
# 2. **Studio/platform** meminimalkan risiko flop dengan memprediksi tema dan genre yang berpotensi laku.  
# 
# **Referensi Terkait**:  
# - [The Netflix Recommender System: Algorithms, Business Value, and Innovation](https://ailab-ua.github.io/courses/resources/netflix_recommender_system_tmis_2015.pdf)  
# - [A Survey of Movie Recommendation Techniques](https://medium.com/@akshaymouryaart/a-survey-on-movie-recommendation-system-d9610777f8e5)  

# ### 1.Import Library
# 
# Kode pada cell ini mengimpor pustaka-pustaka yang diperlukan untuk analisis data, pemrosesan teks, dan visualisasi. Pustaka tersebut mencakup `pandas`, `numpy` dan `scipy` untuk manipulasi data, `json` untuk pengolahan data JSON, `scikit-learn` untuk ekstraksi fitur teks dan penghitungan kesamaan, serta `matplotlib` untuk visualisasi data. Semua pustaka ini digunakan untuk mendukung proses analisis dan pengembangan sistem rekomendasi.

# In[24]:


import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns


# ## 2. Data Understanding

# ### 2.1 Memuat Dataset
# 
# Kode berikut digunakan untuk memuat dataset TMDB Movie ke dalam sebuah DataFrame menggunakan pustaka `pandas`. Dataset ini disimpan dalam file CSV bernama `tmdb_5000_movies.csv`. Setelah dataset dimuat, lima baris pertama dari dataset ditampilkan menggunakan fungsi `head()`. Hal ini bertujuan untuk memahami struktur data, termasuk kolom-kolom yang tersedia dan beberapa nilai awalnya.

# In[2]:


# Load the dataset
movies = pd.read_csv('./tmdb_5000_movies.csv')

# Tampilkan 5 data teratas
movies.head()


# ### 2.2 Analisis Kualitas Data

# Kode berikut digunakan untuk menganalisis kualitas dataset dengan cara:
# 1. **`movies.info()`**: Menampilkan informasi umum tentang dataset, termasuk jumlah baris, kolom, tipe data, dan jumlah nilai non-null di setiap kolom.
# 2. **`movies.isnull().sum()`**: Menghitung jumlah nilai yang hilang (missing values) di setiap kolom.
# 3. **`movies.duplicated().sum()`**: Menghitung jumlah baris duplikat dalam dataset.
# 
# #### Insight dari Output
# 1. **Informasi Dataset**:
#     - Dataset memiliki **4803 baris** dan **20 kolom**.
#     - Kolom memiliki berbagai tipe data, termasuk `int64`, `float64`, dan `object`.
#     - Beberapa kolom memiliki nilai yang hilang, seperti `homepage` (3091 nilai hilang), `overview` (3 nilai hilang), `release_date` (1 nilai hilang), `runtime` (2 nilai hilang), dan `tagline` (844 nilai hilang).
#     - Tidak ada baris duplikat dalam dataset.
# 
# 2. **Kapasitas Memori**:
#     - Dataset menggunakan **750.6 KB** memori.
# 
# 3. **Kualitas Data**:
#     - Kolom seperti `homepage` dan `tagline` memiliki banyak nilai yang hilang, sehingga perlu dipertimbangkan apakah kolom ini relevan untuk analisis atau perlu dihapus.
#     - Kolom lain seperti `overview`, `release_date`, dan `runtime` memiliki sedikit nilai yang hilang, sehingga dapat diisi (imputasi) atau dihapus barisnya.

# In[3]:


# tampilkan informasi dataset
movies.info()


# In[4]:


# missing values dan duplicate values
print("Missing values:")
print(movies.isnull().sum())
print("\nDuplicate values:")
print(movies.duplicated().sum())


# ### 2.3 Exploratory Data Analysis (EDA)
# 
# #### Distribusi Genre Film
# 
# Kode di atas digunakan untuk menganalisis distribusi genre film dalam dataset. Berikut adalah langkah-langkah yang dilakukan:
# 
# 1. **Ekstraksi Genre**:
#     - Kolom `genres` yang berisi data dalam format JSON diubah menjadi daftar nama genre menggunakan fungsi `json.loads` dan `lambda`.
#     - Hasilnya disimpan dalam kolom baru bernama `genres_list`.
# 
# 2. **Menggabungkan Semua Genre**:
#     - Semua genre dari setiap film digabungkan menjadi satu daftar menggunakan list comprehension, menghasilkan variabel `all_genres`.
# 
# 3. **Menghitung Frekuensi Genre**:
#     - Frekuensi kemunculan setiap genre dihitung menggunakan `pd.Series` dan `value_counts()`.
#     - Hanya 10 genre teratas yang diambil untuk analisis lebih lanjut.
# 
# 4. **Visualisasi**:
#     - Data frekuensi genre divisualisasikan dalam bentuk diagram batang menggunakan `matplotlib`.
#     - Diagram ini menunjukkan 10 genre teratas berdasarkan jumlah kemunculannya dalam dataset.
# 
# #### Output Visualisasi
# 
# Diagram batang yang dihasilkan menunjukkan distribusi 10 genre teratas dalam dataset. Genre dengan jumlah kemunculan tertinggi adalah **Drama**, diikuti oleh **Comedy**, **Thriller**, dan genre lainnya. Visualisasi ini memberikan gambaran umum tentang genre yang paling sering muncul dalam dataset.

# In[5]:


# Exploratory Data Analysis (EDA)
# Tampilkan distribusi Genre
movies['genres_list'] = movies['genres'].apply(json.loads).apply(lambda x: [d['name'] for d in x])
all_genres = [g for sub in movies['genres_list'] for g in sub]
genre_counts = pd.Series(all_genres).value_counts().head(10)

plt.figure(figsize=(8,5))
genre_counts.plot(kind='bar')
plt.title('Top 10 Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# ## 3. Data Preparation

# ### 3.1 Ekstraksi dan Transformasi Data
# 
# Kode berikut digunakan untuk mempersiapkan data sebelum digunakan dalam analisis atau pengembangan model. Berikut adalah langkah-langkah yang dilakukan:
# 
# 1. **Salin Dataset**:
#     - Dataset `movies` disalin ke dalam variabel baru bernama `movies_prep` menggunakan fungsi `copy()`. Hal ini dilakukan untuk menjaga dataset asli tetap utuh dan tidak terpengaruh oleh perubahan yang dilakukan pada dataset baru.
# 
# 2. **Ekstraksi Kolom `keywords`**:
#     - Kolom `keywords` yang berisi data dalam format JSON diubah menjadi daftar kata kunci menggunakan fungsi `json.loads` dan `lambda`.
#     - Setiap elemen JSON dalam kolom `keywords` diekstrak menjadi daftar nama kata kunci (`keywords_list`).
# 
# 3. **Penanganan Nilai Kosong pada Kolom `overview`**:
#     - Nilai kosong (missing values) pada kolom `overview` diisi dengan string kosong (`''`) menggunakan fungsi `fillna()`. Hal ini memastikan bahwa kolom `overview` tidak memiliki nilai kosong yang dapat menyebabkan error saat digunakan.
# 
# 4. **Seleksi Kolom Penting**:
#     - Dataset disederhanakan dengan hanya menyimpan kolom-kolom yang relevan, yaitu:
#       - `title`: Judul film.
#       - `genres_list`: Daftar genre film.
#       - `keywords_list`: Daftar kata kunci terkait film.
#       - `overview`: Ringkasan cerita film.
#       - `popularity`: Skor popularitas film.
# 
# 5. **Output Dataset**:
#     - Dataset hasil transformasi ditampilkan untuk memastikan bahwa proses persiapan data telah berhasil dilakukan.
# 
# #### Output
# Dataset `movies_prep` yang dihasilkan berisi informasi yang telah diproses dan siap digunakan untuk analisis lebih lanjut, seperti pembuatan sistem rekomendasi

# In[6]:


movies_prep = movies.copy()
# ekstrak keywords sebagai list
movies_prep['keywords_list'] = movies_prep['keywords'].apply(json.loads).apply(lambda x: [d['name'] for d in x])
movies_prep['overview'] = movies_prep['overview'].fillna('')

movies_prep = movies_prep[['title','genres_list','keywords_list','overview','popularity']]
movies_prep


# ### 3.2 Feature Engineering

# Kode berikut digunakan untuk melakukan rekayasa fitur (feature engineering) pada dataset `movies_prep`. Langkah-langkah yang dilakukan adalah sebagai berikut:
# 
# 1. **Binarisasi Genre**:
#     - `MultiLabelBinarizer` digunakan untuk mengubah daftar genre dalam kolom `genres_list` menjadi representasi matriks biner.
#     - Setiap genre direpresentasikan sebagai kolom, dengan nilai `1` jika film memiliki genre tersebut, dan `0` jika tidak.
#     - Hasilnya disimpan dalam variabel `genre_mat`.
# 
# 2. **Ekstraksi Fitur dari Overview**:
#     - `TfidfVectorizer` digunakan untuk mengubah teks dalam kolom `overview` menjadi representasi numerik berbasis TF-IDF (Term Frequency-Inverse Document Frequency).
#     - Parameter `stop_words='english'` digunakan untuk menghapus kata-kata umum dalam bahasa Inggris, dan `max_features=5000` membatasi jumlah fitur maksimum menjadi 5000.
#     - Hasilnya disimpan dalam variabel `over_mat`.
# 
# 3. **Ekstraksi Fitur dari Keywords**:
#     - `TfidfVectorizer` juga digunakan untuk memproses kolom `keywords_list`, yang berisi daftar kata kunci terkait film.
#     - Daftar kata kunci digabungkan menjadi string menggunakan fungsi `lambda`, kemudian diubah menjadi representasi numerik berbasis TF-IDF.
#     - Parameter `max_features=3000` membatasi jumlah fitur maksimum menjadi 3000.
#     - Hasilnya disimpan dalam variabel `key_mat`.
# 
# 4. **Penggabungan Semua Fitur**:
#     - Matriks fitur dari genre (`genre_mat`), overview (`over_mat`), keywords (`key_mat`), dan popularitas (`movies_prep[['popularity']].values`) digabungkan menjadi satu matriks fitur menggunakan fungsi `hstack`.
#     - Matriks hasil penggabungan ini disimpan dalam variabel `feature_mat` dan akan digunakan sebagai input untuk model rekomendasi.
# 
# #### Output
# Matriks fitur `feature_mat` yang dihasilkan adalah representasi numerik dari data film, mencakup informasi genre, overview, keywords, dan popularitas. Matriks ini siap digunakan untuk analisis lebih lanjut, seperti penghitungan kesamaan atau pembuatan model rekomendasi.

# In[7]:


mlb = MultiLabelBinarizer()
genre_mat = mlb.fit_transform(movies_prep['genres_list'])

tfidf_over = TfidfVectorizer(stop_words='english', max_features=5000)
over_mat = tfidf_over.fit_transform(movies_prep['overview'])

tfidf_key = TfidfVectorizer(stop_words='english', max_features=3000)
key_mat = tfidf_key.fit_transform(movies_prep['keywords_list'].apply(lambda x:' '.join(x)))

feature_mat = hstack([genre_mat, over_mat, key_mat, movies_prep[['popularity']].values])


# ## 4. Modeling and Results  
# ### 4.1 Content‑based Filtering

# Kode berikut digunakan untuk mengimplementasikan sistem rekomendasi berbasis konten (content-based filtering). Sistem ini merekomendasikan film berdasarkan kesamaan fitur dengan film yang dipilih pengguna.
# 
# #### Langkah-langkah Implementasi
# 
# 1. **Menghitung Similarity Matrix**:
#     - `cos_sim = cosine_similarity(feature_mat, feature_mat)`:
#       Matriks kesamaan dihitung menggunakan *cosine similarity* antara semua pasangan film dalam dataset. Matriks ini berbentuk persegi dengan ukuran `n x n`, di mana `n` adalah jumlah film dalam dataset. Nilai dalam matriks menunjukkan tingkat kesamaan antara dua film.
# 
# 2. **Membuat Indeks Film**:
#     - `indices = pd.Series(movies_prep.index, index=movies_prep['title']).drop_duplicates()`:
#       Sebuah *Series* dibuat untuk memetakan judul film ke indeksnya dalam dataset. Hal ini mempermudah pencarian indeks berdasarkan judul film.
# 
# 3. **Fungsi `get_recommendations`**:
#     - Fungsi ini digunakan untuk mendapatkan rekomendasi film berdasarkan judul film yang diberikan.
#     - **Parameter**:
#       - `title`: Judul film yang digunakan sebagai referensi.
#       - `top_n`: Jumlah rekomendasi film yang diinginkan (default: 10).
#     - **Langkah-langkah**:
#       - `idx = indices[title]`: Mendapatkan indeks film berdasarkan judulnya.
#       - `sims = list(enumerate(cos_sim[idx]))`: Mengambil daftar kesamaan antara film referensi dan semua film lainnya.
#       - `sims = sorted(sims, key=lambda x: x[1], reverse=True)[1:top_n+1]`: Mengurutkan daftar kesamaan secara menurun dan mengambil `top_n` film teratas (mengabaikan film itu sendiri).
#       - `rec_idx = [i for i,_ in sims]`: Mendapatkan indeks film yang direkomendasikan.
#       - `return movies_prep['title'].iloc[rec_idx]`: Mengembalikan judul film yang direkomendasikan.
# 
# 4. **Contoh Penggunaan**:
#     - `get_recommendations('Avatar', 10)`:
#       Fungsi ini dipanggil untuk mendapatkan 10 rekomendasi film yang mirip dengan "Avatar".
# 
# #### Output
# Kode ini akan mencetak daftar 10 film yang direkomendasikan berdasarkan kesamaan fitur dengan film "Avatar".

# In[20]:


cos_sim = cosine_similarity(feature_mat, feature_mat)
indices = pd.Series(movies_prep.index, index=movies_prep['title']).drop_duplicates()

def get_recommendations(title, top_n=10):
    idx = indices[title]
    sims = list(enumerate(cos_sim[idx]))
    sims = sorted(sims, key=lambda x: x[1], reverse=True)[1:top_n+1]
    rec_idx = [i for i, _ in sims]
    recommendations = movies_prep[['title', 'genres_list']].iloc[rec_idx].copy()

    return recommendations

# Tampilkan genre dari Avatar untuk perbandingan
avatar_genres = movies_prep.loc[indices['Avatar'], 'genres_list']

# Contoh output
print(f"Genre untuk 'Avatar': {avatar_genres}")
print("\nRekomendasi untuk 'Avatar':")
print(get_recommendations('Avatar', 10))


# ## 5. Evaluation

# ### Menghitung Skor Similarity untuk Film "Avatar"
# 
# Kode berikut digunakan untuk menghitung skor kesamaan (*similarity scores*) antara film **Avatar** dengan film lainnya dalam dataset. Skor kesamaan dihitung menggunakan matriks kesamaan (*cosine similarity matrix*) yang telah dibuat sebelumnya.
# 
# #### Penjelasan Kode
# 1. **Mengambil Indeks Film "Avatar"**:
#     - `avatar_idx = indices['Avatar']`:
#       Mendapatkan indeks film "Avatar" dari *Series* `indices` yang memetakan judul film ke indeksnya dalam dataset.
# 
# 2. **Menghitung Skor Kesamaan**:
#     - `sim_scores = list(enumerate(cos_sim[avatar_idx]))`:
#       Mengambil skor kesamaan antara film "Avatar" dan semua film lainnya dari matriks kesamaan `cos_sim`.
#     - `sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]`:
#       Mengurutkan skor kesamaan secara menurun dan mengambil 10 film teratas yang paling mirip dengan "Avatar" (mengabaikan film itu sendiri).
# 
# 3. **Ekstraksi Skor dan Judul Film**:
#     - `similarity_scores = [score for idx, score in sim_scores]`:
#       Menyimpan skor kesamaan dari 10 film teratas ke dalam daftar `similarity_scores`.
#     - `movie_titles = movies_prep['title'].iloc[[idx for idx, score in sim_scores]].values`:
#       Mengambil judul film dari dataset `movies_prep` berdasarkan indeks film yang telah dipilih.
# 
# #### Output
# Hasil dari kode ini adalah daftar skor kesamaan (*similarity scores*) untuk 10 film yang paling mirip dengan "Avatar". Berikut adalah skor kesamaan yang dihasilkan:
# 

# In[ ]:


# Dapatkan skor similarity untuk film Avatar
avatar_idx = indices['Avatar']
sim_scores = list(enumerate(cos_sim[avatar_idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]  # Top 10

# Ekstrak skor dan judul
similarity_scores = [score for idx, score in sim_scores]
movie_titles = movies_prep['title'].iloc[[idx for idx, score in sim_scores]].values


# ### Penjelasan Fungsi Kode
# 
# Kode ini digunakan untuk membuat visualisasi berupa diagram batang (*bar plot*) yang menunjukkan distribusi skor kesamaan (*cosine similarity*) antara film **Avatar** dan 10 film yang direkomendasikan.
# 
# #### Penjelasan Langkah-langkah:
# 1. **Membuat Figure dan Axis**:
#     ```python
#     plt.figure(figsize=(12, 6))
#     ax = sns.barplot(x=movie_titles, y=similarity_scores, palette="viridis")
#     ```
#     - `plt.figure(figsize=(12, 6))`: Membuat kanvas untuk plot dengan ukuran 12x6 inci.
#     - `sns.barplot(...)`: Membuat diagram batang menggunakan `movie_titles` sebagai sumbu-x dan `similarity_scores` sebagai sumbu-y. Palet warna yang digunakan adalah "viridis".
# 
# 2. **Menambahkan Judul dan Label**:
#     ```python
#     plt.title('Distribusi Skor Similarity untuk Rekomendasi Avatar', fontsize=16, pad=20)
#     plt.xlabel('Film yang Direkomendasikan', fontsize=12)
#     plt.ylabel('Skor Cosine Similarity', fontsize=12)
#     plt.xticks(rotation=45, ha='right', fontsize=10)
#     plt.ylim(0.7, 0.95)
#     ```
#     - `plt.title(...)`: Menambahkan judul pada plot.
#     - `plt.xlabel(...)` dan `plt.ylabel(...)`: Menambahkan label pada sumbu-x dan sumbu-y.
#     - `plt.xticks(...)`: Memutar label pada sumbu-x sebesar 45 derajat agar lebih mudah dibaca.
#     - `plt.ylim(...)`: Mengatur batas nilai pada sumbu-y antara 0.7 hingga 0.95.
# 
# 3. **Menambahkan Nilai di Atas Setiap Bar**:
#     ```python
#     for p in ax.patches:
#          ax.annotate(f"{p.get_height():.2f}", 
#                          (p.get_x() + p.get_width() / 2., p.get_height()), 
#                          ha='center', va='center', 
#                          xytext=(0, 10), 
#                          textcoords='offset points',
#                          fontsize=10)
#     ```
#     - Loop ini menambahkan anotasi berupa nilai skor kesamaan di atas setiap batang pada diagram.
# 
# 4. **Menambahkan Garis Horizontal untuk Rata-rata**:
#     ```python
#     ax.axhline(y=np.mean(similarity_scores), color='red', linestyle='--', linewidth=1)
#     ax.text(x=len(movie_titles)-1, y=np.mean(similarity_scores)+0.01, 
#               s=f'Rata-rata: {np.mean(similarity_scores):.2f}', 
#               color='red', fontsize=10)
#     ```
#     - `ax.axhline(...)`: Menambahkan garis horizontal pada nilai rata-rata skor kesamaan.
#     - `ax.text(...)`: Menambahkan teks "Rata-rata" di dekat garis horizontal.
# 
# 5. **Menyesuaikan Layout dan Menyimpan Gambar**:
#     ```python
#     plt.tight_layout()
#     plt.savefig('similarity_scores_distribution.png', dpi=300, bbox_inches='tight')
#     plt.show()
#     ```
#     - `plt.tight_layout()`: Mengatur tata letak agar elemen plot tidak saling tumpang tindih.
#     - `plt.savefig(...)`: Menyimpan plot sebagai file gambar dengan nama `similarity_scores_distribution.png`.
#     - `plt.show()`: Menampilkan plot.
# 
# ---
# 
# ### Output
# Output dari kode ini adalah diagram batang seperti berikut:
# 
# - **Sumbu-x**: Judul film yang direkomendasikan.
# - **Sumbu-y**: Skor kesamaan (*cosine similarity*) dengan film **Avatar**.
# - **Batang**: Menunjukkan skor kesamaan untuk setiap film.
# - **Anotasi**: Nilai skor kesamaan di atas setiap batang.
# - **Garis Horizontal**: Menunjukkan rata-rata skor kesamaan dengan teks "Rata-rata" di dekatnya.

# In[25]:


# Buat figure dan axis
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=movie_titles, y=similarity_scores, palette="viridis")

# Atur judul dan label
plt.title('Distribusi Skor Similarity untuk Rekomendasi Avatar', fontsize=16, pad=20)
plt.xlabel('Film yang Direkomendasikan', fontsize=12)
plt.ylabel('Skor Cosine Similarity', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.ylim(0.7, 0.95)

# Tambahkan nilai di atas setiap bar
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize=10)

# Tambahkan garis horizontal
ax.axhline(y=np.mean(similarity_scores), color='red', linestyle='--', linewidth=1)
ax.text(x=len(movie_titles)-1, y=np.mean(similarity_scores)+0.01, 
        s=f'Rata-rata: {np.mean(similarity_scores):.2f}', 
        color='red', fontsize=10)

# Adjust layout
plt.tight_layout()

# Simpan gambar
plt.savefig('similarity_scores_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


# ### Analisis Visualisasi Distribusi Skor Cosine Similarity untuk Rekomendasi “Avatar”
# 
# Dari visualisasi **Distribusi Skor Cosine Similarity** untuk rekomendasi “Avatar”, dapat disimpulkan beberapa hal berikut:
# 
# 1. **Skor Sangat Tinggi**:
#     - Semua film rekomendasi memiliki skor yang sangat tinggi (sekitar **0.90–0.95**).
#     - Hal ini menandakan bahwa fitur konten (genre, sinopsis, keywords, popularity) dari film‑film tersebut benar‑benar mirip dengan “Avatar”.
# 
# 2. **Rentang Skor yang Sempit**:
#     - Rentang skor yang sempit memperlihatkan bahwa **Top‑10 rekomendasi terkelompok rapat di “zona kemiripan tinggi”**.
#     - Model sangat yakin bahwa semua film ini relevan, tetapi cenderung kurang memberikan variasi dalam rekomendasi.
# 
# 3. **Rata‑rata Skor Mendekati 1.0**:
#     - Rata‑rata skor yang mendekati **1.0** menunjukkan bahwa algoritma berhasil menangkap kesamaan semantik dengan kuat.
#     - Namun, hal ini juga bisa menandakan adanya risiko **overfitting** pada fitur metadata yang digunakan.
# 
# Kesimpulan ini memberikan wawasan tentang performa model rekomendasi berbasis konten, sekaligus menunjukkan potensi area untuk perbaikan, seperti meningkatkan variasi rekomendasi.
