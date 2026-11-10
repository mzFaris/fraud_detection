// Laporan Tugas Machine Learning
// Sumber tunggal laporan ini adalah berkas Typst ini.
// Kompilasi: tinymist compile LAPORAN.typ LAPORAN.pdf

#set document(
  title: "Klasifikasi Fraud Transaksi Kartu Kredit dengan Random Forest dan SMOTE",
)

#set page(
  numbering: "1",
)

#set text(
  lang: "id",
)

#set par(justify: true)
#set heading(numbering: "1.")

#show raw: set text(font: "DejaVu Sans Mono")
#show raw.where(block: true): set block(
  fill: luma(246),
  stroke: 0.5pt + luma(200),
  inset: 10pt,
  radius: 3pt,
  width: 100%,
)

#show figure.caption: set text(size: 9pt)

// ---------------------------------------------------------------- judul

#align(center)[
  #text(size: 18pt, weight: "bold")[
    Laporan Tugas Machine Learning
  ]
  #v(0.4em)
  #text(size: 14pt)[
    Klasifikasi Fraud Transaksi Kartu Kredit \
    dengan Random Forest dan SMOTE
  ]
]

#v(0.6em)
#line(length: 100%, stroke: 0.8pt + luma(120))
#v(0.4em)

// ---------------------------------------------------------------- 1

= Metode yang Digunakan

Kasus yang dipilih adalah *klasifikasi fraud transaksi kartu kredit*, yaitu
memisahkan transaksi fraud (`Class = 1`) dari transaksi normal (`Class = 0`).
Data yang dipakai adalah dataset _Credit Card Fraud Detection_ dari Kaggle:
284.807 transaksi pemegang kartu Eropa selama dua hari, dengan 492 transaksi
fraud atau hanya *0,172%* dari keseluruhan data. Fitur `V1` sampai `V28` sudah
merupakan hasil transformasi PCA, `Amount` adalah nominal transaksi, dan `Time`
adalah detik yang berlalu sejak transaksi pertama.

Metode yang digunakan:

- *Random Forest Classifier* dengan 200 pohon dan `max_depth=10` sebagai model
  klasifikasi.
- *SMOTE* (_Synthetic Minority Over-sampling Technique_) untuk menyeimbangkan
  kelas fraud pada data train.
- *Dua skenario* yang dibandingkan pada test set yang sama: *baseline* (tanpa
  SMOTE) dan *SMOTE* (data train diseimbangkan lebih dulu).
- *AUPRC* (average precision) sebagai metrik utama, didampingi accuracy,
  precision, recall, F1, ROC-AUC, dan confusion matrix.

// ---------------------------------------------------------------- 2

= Alasan Memilih Metode

- *Random Forest* cocok untuk data tabular, tahan terhadap outlier, dan mampu
  menangkap interaksi antarfitur tanpa scaling tambahan pada fitur PCA.
- *SMOTE* dipilih karena kelas fraud sangat sedikit, hanya 394 contoh di data
  train. SMOTE membuat sampel sintetis dari tetangga terdekat kelas minoritas,
  sehingga model tidak hanya belajar dari segelintir contoh.
- *`max_depth=10`* membatasi kompleksitas tiap pohon sebagai regularisasi.
  Tanpa batas kedalaman, Random Forest dapat menghafal data train dan
  overfitting.
- *AUPRC* dipilih karena dataset sangat tidak seimbang: accuracy mudah terlihat
  tinggi karena 99,83% transaksi normal, sedangkan AUPRC menilai kemampuan
  menemukan kelas fraud.

// ---------------------------------------------------------------- 3

= Proses Penerapan

Penerapan dilakukan berurutan:

+ *Validasi data*: memastikan kolom wajib tersedia, tidak ada missing value, dan
  `Class` hanya berisi 0 dan 1.
+ *Pemilihan fitur*: `Time` dibuang karena nilainya hanya detik relatif terhadap
  transaksi pertama, bukan waktu kalender yang stabil. Fitur yang dipakai adalah
  `V1` sampai `V28` dan `Amount`.
+ *Stratified train/test split* 80/20 dengan `random_state=42` menghasilkan 227.845 baris train (227.451 non-fraud + 394 fraud) dan 56.962 baris test (56.864 non-fraud + 98 fraud). Test set memuat 98 transaksi fraud (0,1720%) sehingga proporsi kelas tetap terjaga.
+ *Scaling*: `Amount` ditransformasi dengan `RobustScaler` yang di-fit hanya pada
  data train, lalu dipakai untuk mentransformasi data test.
+ *SMOTE hanya diterapkan pada data train*. Data train yang semula berisi 227.451 non-fraud dan 394 fraud (total 227.845 baris) diseimbangkan menjadi 227.451 non-fraud dan 227.451 fraud sintetis, sehingga total data train setelah oversampling menjadi 454.902 baris. Dengan demikian SMOTE menambah 227.057 sampel sintetis (227.451 − 394). Data test tidak pernah di-resample agar evaluasi tetap mencerminkan data asli.
+ *Training*: baseline dilatih pada data train yang masih imbalanced, model kedua
  dilatih pada data train hasil SMOTE, keduanya dengan parameter yang sama.
+ *Evaluasi*: kedua model diuji pada test set asli yang sama.

Scaling dan SMOTE sengaja dilakukan setelah split untuk mencegah *data leakage*.

Parameter model yang dipakai:

```text
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
```

Evaluasi memakai tujuh ukuran. *Accuracy* adalah proporsi prediksi benar secara
keseluruhan. *Precision* menyatakan dari transaksi yang diprediksi fraud, berapa
yang benar-benar fraud. *Recall* menyatakan dari transaksi fraud yang sebenarnya,
berapa yang berhasil ditemukan. *F1* adalah harmonic mean precision dan recall.
*ROC-AUC* mengukur kemampuan memisahkan kedua kelas pada berbagai threshold.
*AUPRC* mengukur area di bawah precision-recall curve dan menjadi metrik utama
untuk data yang sangat imbalanced. *Confusion matrix* merinci jumlah true
negative, false positive, false negative, dan true positive.

// ---------------------------------------------------------------- 4

= Hasil

== Perbandingan metrik

#figure(
  table(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    inset: (x: 6pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 0.9pt),
    table.header([*Skenario*], [*Accuracy*], [*Precision*], [*Recall*], [*F1*], [*ROC-AUC*], [*AUPRC*]),
    table.hline(stroke: 0.5pt),
    [Baseline], [0,9996], [0,9405], [0,8061], [0,8681], [0,9732], [*0,8679*],
    [SMOTE], [0,9976], [0,4019], [0,8571], [0,5472], [0,9796], [0,8065],
    table.hline(stroke: 0.9pt),
  ),
  caption: [Metrik kedua skenario pada test set (56.962 transaksi, 98 fraud).
    Nilai AUPRC ditebalkan karena merupakan metrik utama.],
) <tab:metrik>

#figure(
  table(
    columns: 5,
    align: (left, right, right, right, right),
    inset: (x: 6pt, y: 5pt),
    stroke: none,
    table.hline(stroke: 0.9pt),
    table.header([*Skenario*], [*TN*], [*FP*], [*FN*], [*TP*]),
    table.hline(stroke: 0.5pt),
    [Baseline], [56.859], [5], [19], [79],
    [SMOTE], [56.739], [125], [14], [84],
    table.hline(stroke: 0.9pt),
  ),
  caption: [Confusion matrix pada test set. FN berarti transaksi fraud yang lolos
    dari deteksi; FP berarti transaksi normal yang ditandai fraud.],
) <tab:cm>

#figure(
  image("outputs/metric_comparison.png", width: 74%),
  caption: [Perbandingan seluruh metrik antara baseline dan SMOTE.],
) <fig:perbandingan>

#figure(
  image("outputs/precision_recall_curve.png", width: 64%),
  caption: [Precision-recall curve kedua model pada test set.],
) <fig:prcurve>

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  figure(
    image("outputs/confusion_matrix_baseline.png", width: 100%),
    caption: [Baseline: 19 FN, 5 FP.],
  ),
  figure(
    image("outputs/confusion_matrix_smote.png", width: 100%),
    caption: [SMOTE: 14 FN, 125 FP.],
  ),
)

== Tafsir hasil

Baseline tanpa penyeimbangan sudah sangat kuat: AUPRC 0,8679, precision 0,9405,
recall 0,8061. Dari 98 transaksi fraud di test set, 79 tertangkap dan 19 lolos,
sementara hanya 5 transaksi normal yang salah ditandai. Pada threshold 0,5 model
sudah seimbang antara menangkap fraud dan menekan alarm palsu, meskipun data
train hanya berisi 394 contoh fraud.

SMOTE menambah 227.057 sampel sintetis sehingga data train menjadi seimbang,
tetapi efeknya pada test set bercabang. Recall naik dari 0,8061 menjadi 0,8571
karena false negative turun dari 19 menjadi 14, namun precision jatuh dari
0,9405 menjadi 0,4019 karena false positive melonjak dari 5 menjadi 125. F1
turun ke 0,5472 dan AUPRC sebagai metrik utama turun dari 0,8679 menjadi 0,8065.

Penyebab yang paling mungkin adalah pergeseran distribusi data train. Setelah
oversampling, proporsi fraud di data train menjadi 50%, jauh di atas proporsi
aslinya (0,17%), sehingga model cenderung memberi skor fraud yang lebih tinggi
pada transaksi di test set dan threshold 0,5 menjadi terlalu longgar. Selain
itu, sampel sintetis SMOTE dibuat dengan interpolasi antartetangga kelas fraud,
sehingga sebagian kemungkinan berada di daerah batas keputusan yang juga berisi
transaksi normal. Akibatnya model lebih agresif menandai fraud: lima fraud
tambahan tertangkap (FN turun dari 19 menjadi 14) dengan harga 120 alarm palsu
tambahan, dan lonjakan itu menekan precision di region recall tinggi, area yang
paling menentukan AUPRC (lihat @fig:prcurve). Dugaan ini belum diuji secara
langsung dalam eksperimen ini.

ROC-AUC justru naik sedikit, dari 0,9732 menjadi 0,9796. Kedua metrik tidak
bertentangan karena mengukur hal yang berbeda. ROC-AUC memakai _false positive
rate_, yaitu FP dibagi seluruh 56.864 transaksi normal. Pada threshold 0,5,
tambahan 120 FP hanya menaikkan rasio itu dari sekitar 0,009% menjadi 0,22%,
sehingga bentuk kurva ROC nyaris tidak berubah. AUPRC memakai precision, yang
penyebutnya hanya transaksi yang diprediksi fraud (TP + FP). Karena fraud
sangat langka, FP yang kecil terhadap populasi normal menjadi besar terhadap
jumlah prediksi positif: precision turun dari 79/84 (0,9405) menjadi 84/209
(0,4019). Dengan kata lain, ROC-AUC meredam kesalahan pada kelas mayoritas,
sedangkan AUPRC langsung mencerminkan beban alarm palsu. Untuk kasus fraud,
AUPRC lebih mencerminkan kebutuhan operasional.

== Mengapa accuracy tidak memadai

Sekitar 99,83% transaksi adalah non-fraud, sehingga model yang menebak *semua
transaksi sebagai non-fraud* pun mendapat accuracy 99,83% tanpa menangkap satu
fraud pun. Gejala itu terlihat di @tab:metrik: kedua model sama-sama di atas
99,7% accuracy dengan selisih hanya 0,0020, padahal precision keduanya berbeda
jauh (0,9405 berbanding 0,4019). Accuracy nyaris tidak membedakan kualitas model,
sehingga penilaian harus bertumpu pada AUPRC dan confusion matrix, terutama
false negative, karena setiap fraud yang lolos adalah kerugian langsung.


== Keterbatasan

Perbandingan ini memakai satu pembagian data (`random_state=42`) dan satu kali
pelatihan per skenario, dengan hanya 98 transaksi fraud di test set. Pada
jumlah sampel sekecil itu, selisih satu atau dua transaksi dapat menggeser
recall dan precision cukup jauh, sehingga selisih AUPRC 0,0614 dan selisih FN
(19 berbanding 14) perlu dibaca sebagai indikasi, bukan bukti kuat. Evaluasi
yang lebih meyakinkan dapat memakai _stratified k-fold_ atau beberapa nilai
`random_state` dengan melaporkan rata-rata dan simpangan bakunya. Selain itu,
kedua model memakai hiperparameter yang sama dan threshold tetap 0,5; hasil
SMOTE mungkin berbeda bila threshold dan parameter dituning terpisah, atau bila
dibandingkan dengan `class_weight="balanced"`.

= Kesimpulan

+ Random Forest baseline pada data imbalanced mencapai performa kuat: AUPRC
  0,8679, precision 0,9405, recall 0,8061, dengan hanya 19 fraud yang lolos dari
  98 transaksi fraud di test set.
+ SMOTE hanya diterapkan pada data train sehingga tidak ada data leakage, dan
  secara mekanis bekerja: data train seimbang dan recall naik ke 0,8571 (false
  negative 19 menjadi 14).
+ Kenaikan recall itu dibayar mahal: false positive melonjak dari 5 menjadi 125,
  precision jatuh ke 0,4019, dan AUPRC turun menjadi 0,8065. Pada split ini,
  baseline lebih unggul 0,0614 pada metrik utama (lihat keterbatasan di atas).
+ Accuracy tidak memadai karena kedua model sama-sama di atas 99,7% namun
  kualitasnya berbeda jauh; penilaian harus bertumpu pada AUPRC, precision,
  recall, F1, dan confusion matrix.
+ ROC-AUC dan AUPRC dapat berlawanan arah (ROC-AUC sedikit memihak SMOTE, AUPRC
  memihak baseline), sehingga pelaporan tidak boleh bergantung pada satu angka;
  langkah berikutnya adalah tuning threshold dan cost-sensitive learning, bukan
  sekadar menambah oversampling.
