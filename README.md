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
