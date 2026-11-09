# Fraud Detection - Credit Card Transactions

Tugas Machine Learning: klasifikasi fraud transaksi kartu kredit menggunakan
Random Forest, dengan fokus pada **imbalanced data** dan **SMOTE**.

Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284.807 transaksi, 492 fraud, 0,172% fraud).

## Menyiapkan environment

Python dan seluruh dependency Python dikelola oleh `uv`. Tidak ada dev shell
atau virtual environment tambahan yang perlu diaktifkan secara manual.

Instal `uv` jika belum tersedia (lihat
[dokumentasi uv](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Setelah `uv` tersedia, siapkan environment Python:

```bash
uv sync
```

Perintah ini akan membuat `.venv`, menggunakan Python 3.12 sesuai
`.python-version`, dan menginstal dependency dari `uv.lock`.

Jika memakai mesin low-spec, jangan menjalankan training penuh di mesin lokal;
gunakan Google Colab/Kaggle seperti dijelaskan di bawah.

## Menyiapkan dataset

Unduh `creditcard.csv` dari Kaggle, lalu letakkan di root repo:

```
fraud_detection/
├── creditcard.csv
├── fraud_detection/
└── ...
```

Dataset tidak disertakan di repo karena ukurannya sekitar 144 MB.

## Menjalankan program

Dari root repo:

```bash
uv run python -m fraud_detection --data creditcard.csv --output-dir outputs
```

Opsi lain:

```bash
uv run python -m fraud_detection --help
```

Jika memakai mesin low-spec, jalankan di Google Colab/Kaggle. Unggah repo dan
`creditcard.csv`, lalu:

```bash
pip install uv
uv sync
uv run python -m fraud_detection --data creditcard.csv --output-dir outputs
```

## Menambah dependency

```bash
uv add nama-package
uv lock
```

Untuk memperbarui dependency yang sudah ada:

```bash
uv lock --upgrade
uv sync
```

Program akan:

1. Memuat dan memvalidasi dataset.
2. Melakukan stratified train/test split dan scaling `Amount`.
3. Melatih dua model: baseline tanpa SMOTE dan model dengan SMOTE.
4. Mengevaluasi kedua model pada test set yang sama.
5. Menyimpan metrik dan plot ke `outputs/`.

## Alur program (detail)

Entry point `uv run python -m fraud_detection` akan menjalankan
`fraud_detection/__main__.py` (`__main__.py:117-119`). Seluruh orkestrasi ada di
`main()` (`__main__.py:44`): tidak ada pipeline object, hanya skrip lurus yang
memanggil lima fungsi dari empat modul dengan urutan tetap.

### 1. Parsing argumen (`build_parser`, `__main__.py:12-41`)

| Opsi | Default | Dipakai oleh |
|---|---|---|
| `--data` | `creditcard.csv` | `load_data` |
| `--output-dir` | `outputs` | `save_outputs` |
| `--test-size` | `0.2` | `prepare_data` |
| `--random-state` | `42` | split, SMOTE, RandomForest |

### 2. Memuat dan memvalidasi data (`data.py`)

`load_data(args.data)` (`__main__.py:47`) melakukan:

1. Cek keberadaan file; jika tidak ada → `FileNotFoundError` (`data.py:14`).
2. `pd.read_csv` — 284.807 baris × 31 kolom (`data.py:20`).
3. Cek `REQUIRED_COLUMNS = ["Time", V1..V28, "Amount", "Class"]`; kolom yang
   hilang → `ValueError` (`data.py:22-24`).
4. Cek missing value pada kolom wajib (`data.py:26-29`).
5. Cek `Class` hanya berisi `{0, 1}` dan kedua kelas ada (`data.py:31-35`).

Data masih mentah; belum ada kolom yang dibuang atau ditransformasi. `main`
hanya mencetak shape dan distribusi kelas (`__main__.py:48-55`).

### 3. Preprocessing (`preprocessing.py`)

`prepare_data(...)` (`__main__.py:57`):

1. **Seleksi fitur** (`preprocessing.py:26-27`): `FEATURE_COLUMNS = V1..V28 +
   ["Amount"]`. `Time` sengaja tidak dipakai; `y = Class.astype(int)`.
2. **Stratified split** (`preprocessing.py:29-35`): dengan default menghasilkan
   227.845 baris train dan 56.962 baris test, dengan proporsi fraud 0,172% tetap
   terjaga di kedua bagian.
3. **Scaling** (`preprocessing.py:37-41`): `RobustScaler` di-fit **hanya pada
   `X_train[["Amount"]]`**, lalu diterapkan ke train dan test. `V1..V28` tidak
   di-scale karena sudah hasil PCA. Pola fit-on-train-only ini mencegah leakage.
4. Mengembalikan dataclass immutable `DataSplit` (`preprocessing.py:12-17`).

Test set dibuat **sekali** di sini dan tidak pernah di-resample. Kedua model
melihat `X_test`/`y_test` yang identik, sehingga perbandingan tetap adil.

### 4. Training dua model (`modeling.py`)

Kedua cabang memakai factory estimator yang sama, `_build_model`
(`modeling.py:20-26`):

```python
RandomForestClassifier(n_estimators=200, max_depth=10, random_state, n_jobs=-1)
```

**Baseline** — `train_baseline(...)` (`__main__.py:72`): `model.fit` langsung
pada data imbalanced (`modeling.py:41`); `n_train_before == n_train_after` dan
distribusi kelas dicatat dua kali (`modeling.py:45-48`).

**SMOTE** — `train_with_smote(...)` (`__main__.py:80`):
`SMOTE(...).fit_resample(X_train, y_train)` (`modeling.py:58-59`) membuat sampel
minoritas sintetis dengan interpolasi antar tetangga fraud terdekat, sehingga
train membengkak menjadi sekitar 454.880 baris dengan rasio 50/50. Model baru
dibuat lewat `_build_model` dan di-fit pada `X_resampled, y_resampled`
(`modeling.py:61-62`) — model baseline tidak di-retrain atau diubah.
`n_train_before` (227.845) dan `n_train_after` (±454.880) disimpan di
`TrainingResult` (`modeling.py:64-70`).

### 5. Evaluasi (`evaluation.py`)

`evaluate(name, model, X_test, y_test)` dipanggil sekali per model
(`__main__.py:88-91`). Di dalamnya (`evaluation.py:38-49`) prediksi dihitung
sekali dari test set yang tidak pernah disentuh:

- `model.predict(X_test)` — label 0/1 pada threshold default 0,5.
- `model.predict_proba(X_test)[:, 1]` — probabilitas fraud, dipakai untuk metrik
  yang tidak bergantung threshold.

Metrik yang dihitung: `accuracy`, `precision`, `recall`, `f1` (dengan
`zero_division=0`), `roc_auc` dan `average_precision` (AUPRC) dari `y_proba`,
serta `confusion_matrix` sebagai nested list.

### 6. Menyimpan output (`save_outputs`, `evaluation.py:160`)

1. `mkdir(parents=True, exist_ok=True)` (`evaluation.py:167`).
2. `metrics.json` (`evaluation.py:169-180`): blok `dataset` (`n_test`,
   `fraud_test`, `fraud_rate_test`) plus `results` per nama skenario.
3. `classification_report.txt` (`evaluation.py:182-198`): report per kelas untuk
   masing-masing model.
4. `results_table.md` (`evaluation.py:200-215`): tabel Markdown yang dipakai
   `LAPORAN.typ`.
5. Plot, dengan `matplotlib.use("Agg")` di-set saat import (`evaluation.py:9`)
   agar aman di mesin headless:
   - `_save_confusion_matrices` (`evaluation.py:59-86`) →
     `confusion_matrix_{slug}.png`, dengan `slug = name.lower().replace(" ", "_")`.
   - `_save_precision_recall_curve` (`evaluation.py:89-118`) → kedua kurva dalam
     satu axes, legend berisi AUPRC, plus garis putus-putus pada prevalensi fraud
     sebagai pembanding "random classifier".
   - `_save_metric_comparison` (`evaluation.py:121-157`) → grouped bar chart
     dengan label nilai, sumbu y dibatasi 1,1.
   - Semua helper memanggil `savefig(..., dpi=150, bbox_inches="tight")` lalu
     `plt.close(fig)`.

`save_outputs` mengembalikan `payload`, tetapi `main` mengabaikannya.

### 7. Ringkasan konsol

`__main__.py:95-114` mencetak tabel enam metrik untuk kedua skenario, path output
absolut, dan pengingat bahwa AUPRC adalah metrik utama. `main` mengembalikan `0`,
lalu `__main__.py:119` mengubahnya menjadi exit code proses.

### Ringkasan alur

```
argparse
  → load_data              → DataFrame 284.807×31 yang sudah tervalidasi
      → prepare_data       → stratified split + RobustScaler pada Amount
          ├→ train_baseline    → RF(200, depth 10) pada 227.845 baris imbalanced
          └→ train_with_smote  → SMOTE → ±454.880 baris seimbang → RF(200, depth 10)
              → evaluate (×2 pada X_test/y_test yang sama)
                  → save_outputs
                      → metrics.json, classification_report.txt, results_table.md,
                        confusion_matrix_{baseline,smote}.png,
                        precision_recall_curve.png, metric_comparison.png
                  → print tabel ringkasan
```

### Catatan implementasi

- `TrainingResult.n_train_before/n_train_after` tersimpan tetapi tidak pernah
  dicetak atau ditulis ke disk; `main` hanya mencetak distribusi kelas. Jika
  angka ini ingin masuk laporan, nilainya harus diteruskan ke `payload`.
- `save_outputs` mengembalikan `payload` yang tidak dipakai, sementara tabel
  konsol di `__main__.py:95-109` memformat ulang data yang sama. Jika ada
  perubahan kolom, ubah di kedua tempat (`__main__.py:95-109` dan
  `evaluation.py:200-211`).
- Posisi `matplotlib.use("Agg")` di atas import `pyplot` (`evaluation.py:7-11`)
  bersifat wajib; memindahkannya ke bawah dapat memakai backend GUI dan gagal di
  mesin headless.

## Output

- `outputs/metrics.json` — accuracy, precision, recall, F1, ROC-AUC, AUPRC, confusion matrix.
- `outputs/classification_report.txt` — classification report kedua skenario.
- `outputs/results_table.md` — tabel Markdown siap disalin ke laporan.
- `outputs/confusion_matrix_baseline.png`
- `outputs/confusion_matrix_smote.png`
- `outputs/precision_recall_curve.png`
- `outputs/metric_comparison.png`

Metrik utama adalah **AUPRC (Area Under the Precision-Recall Curve)**, karena
accuracy tidak cukup untuk menilai model pada data yang sangat imbalanced.

Angka pada laporan berasal dari eksekusi di Google Colab (Python 3.13,
scikit-learn 1.6.1, imbalanced-learn 0.14.2). Versi tersebut sedikit berbeda dari
pin di `pyproject.toml`, sehingga nilai dapat bergeser di digit ketiga atau
keempat bila dijalankan ulang dengan environment `uv`.

## Pembahasan

Laporan lengkap ada di [`LAPORAN.typ`](LAPORAN.typ) (sumber tunggal, ditulis
dengan Typst). Untuk menghasilkan PDF:

```bash
tinymist compile LAPORAN.typ LAPORAN.pdf
```

`tinymist` juga berfungsi sebagai LSP Typst, jadi membuka `LAPORAN.typ` di editor
yang mendukung LSP (misalnya Helix) langsung memberi diagnostik dan PDF terbaru
saat file disimpan.
