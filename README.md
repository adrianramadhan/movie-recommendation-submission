# Laporan Proyek Machine Learning – Adrian Putra Ramadhan

## Project Overview
Di era platform streaming dan big‑budget productions, investasi untuk membuat satu film bisa mencapai ratusan juta dolar. Meski begitu, banyak judul blockbuster tetap gagal secara komersial (“box‑office flop”), menimbulkan kerugian finansial besar. Di sisi lain, penonton menghadapi ribuan pilihan judul yang membuat proses memilih film menjadi sulit (paradox of choice). Sistem rekomendasi film membantu:  
1. **Pengguna** menemukan judul yang sesuai minatnya.  
2. **Studio/platform** meminimalkan risiko flop dengan memprediksi tema dan genre yang berpotensi laku.  

Pada proyek ini, kita menggunakan **TMDB 5000 Movie Dataset** (±5.000 film) dengan metadata:  
- Genres  
- Keywords  
- Overview  
- Popularity  

**Pendekatan yang Dipilih**:  
Content‑based Filtering karena:  
- Cepat di‑prototype hanya dengan metadata.  
- Cold‑start friendly: film baru tanpa riwayat interaksi tetap bisa direkomendasikan.  
- Hasil mudah dijelaskan (*“film X direkomendasikan karena genre/narasinya mirip film Y”*).  

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
- Untuk setiap film input atau *user profile* sederhana (list film favorit), pilih film dengan similarity tertinggi.