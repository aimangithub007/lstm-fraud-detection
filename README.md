# LSTM Fraud Detection — Sequential Transaction per User

## 1. Susunan Data

**Struktur:** Transaksi sequential per user (time series), bukan tabel datar biasa.

**Fitur per transaksi (5 kolom):**
```
amount, hour, merchant_category, is_foreign, time_since_last_tx
```

**Label:** `is_fraud` (0/1)

**Cara generate (sintetis):**
- 500 user, masing-masing punya 10–30 transaksi
- Setiap transaksi punya probabilitas **4%** ditandai fraud (`fraud_rate=0.04`)
- Jika fraud → suntik pola mencurigakan: amount 3–8x lipat normal, jam dini hari (00:00–04:00), 70% luar negeri, jeda antar transaksi sangat cepat (<0.4 jam)
- Jika normal → amount sesuai kebiasaan user, jam wajar, mayoritas domestik

**Hasil:** 9.682 transaksi, 416 fraud (**4.3%** — imbalanced tapi cukup untuk belajar)

---

## 2. Skenario Pembuatan Model

1. **EDA** — cek distribusi label, amount, jam, korelasi fitur
2. **Preprocessing** — encode `merchant_category` (LabelEncoder)
3. **Sequence building** — sliding window per user, `window_size=5` (5 transaksi terakhir jadi 1 input), label = transaksi terakhir di window. User dengan riwayat <5 transaksi di-*pre-padding* dengan nol
4. **Split** — 80/20, `stratify` agar proporsi fraud tetap terjaga di train & test
5. **Scaling** — `StandardScaler`, **fit hanya di train** (hindari data leakage)
6. **Class weight** — `class_weight="balanced"` karena imbalanced (bobot fraud ≈ 11.6x non-fraud)
7. **Arsitektur model:**
   ```
   Input → LSTM(64, return_sequences=True) → BatchNorm → Dropout
         → LSTM(32) → BatchNorm → Dropout
         → Dense(32, relu) → Dropout
         → Dense(1, sigmoid)
   ```
8. **Callbacks** — `EarlyStopping` (monitor val_auc), `ReduceLROnPlateau`, `ModelCheckpoint` (simpan model terbaik)
9. **Training** — `binary_crossentropy`, optimizer Adam, max 100 epoch (berhenti otomatis lebih cepat)

---

## 3. Pengujian

**A. Evaluasi pada test set** (20% data, distribusi sama dengan training)
- Metrik: `classification_report`, `confusion_matrix`, ROC-AUC, PR-AUC

**B. Inference manual pada data baru buatan tangan** (3 skenario user)
- **User A** — semua transaksi normal
- **User B** — 4 transaksi normal, transaksi ke-5 mencurigakan (amount tinggi, dini hari, luar negeri, cepat)
- **User C** — user baru, hanya 1 transaksi (uji padding) — langsung mencurigakan

---

## 4. Hasil

### A. Evaluasi test set

| Kelas | Precision | Recall | F1-score |
|---|---|---|---|
| Non-Fraud | 1.00 | 1.00 | 1.00 |
| Fraud | 1.00 | 0.98 | 0.99 |

**Confusion Matrix:**
```
[[1854    0]
 [   2   81]]
```

| Metrik | Nilai |
|---|---|
| ROC-AUC | 0.9997 |
| PR-AUC | 0.9946 (baseline acak hanya 0.0428) |

→ Model menangkap 81 dari 83 fraud asli, tanpa false alarm sama sekali.

### B. Inference data baru

| User | Probabilitas Fraud | Prediksi | Keterangan |
|---|---|---|---|
| A | 0.0002 | ✅ Non-Fraud | Benar — pola normal konsisten |
| B | 0.9999 | 🚨 FRAUD | Benar — anomali relatif terhadap riwayat user |
| C | 0.9998 | 🚨 FRAUD | Benar — padding tetap berfungsi |

---

## Catatan Penting

Hasil ini sangat baik karena data **sintetis** dengan pola fraud yang dibuat jelas. Di data nyata, hasilnya biasanya tidak akan sebagus ini — fraud asli lebih "berisik" dan pola pelaku terus berubah (adversarial). Ini cocok sebagai **demo konsep**, bukan acuan performa produksi.

---

## Bagian Kode yang Spesifik vs Universal

### ❌ Spesifik ke kasus ini (perlu diganti untuk kasus lain)
- `feature_cols` — daftar fitur fraud
- Fungsi `generate_synthetic_transaction_data` — seluruhnya
- `WINDOW_SIZE = 5` — panjang window tergantung domain
- Grouping per `user_id` — tergantung domain (bisa per sensor, per saham, dll)
- `LabelEncoder` untuk `merchant_category`
- `fraud_rate` & `class_weight` — tergantung tingkat imbalance kasus

### ✅ Universal (bisa dipakai ulang)
- Arsitektur LSTM dasar (`LSTM → BatchNorm → Dropout → Dense → sigmoid`)
- Scaling dengan `StandardScaler`
- `train_test_split` dengan `stratify`
- Callbacks (`EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`)
- Evaluasi (`classification_report`, `confusion_matrix`, ROC-AUC, PR-AUC)
- Konsep padding untuk sequence pendek

---

## File Terkait

| File / Folder | Fungsi |
|---|---|
| `lstm-fraud-detection.py` | Training lengkap: EDA → preprocessing → model → evaluasi. Menghasilkan model `.keras` dan folder `-results` |
| `best_lstm_fraud_model.keras` | Model LSTM terlatih (checkpoint terbaik berdasarkan `val_auc`) |
| `inference-fraud-detection.py` | Load model `.keras`, prediksi data transaksi baru (3 skenario uji: normal, anomali, user baru/padding) |
| `lstm-fraud-detection-results/` | Folder gambar hasil EDA & evaluasi dari proses training: <br>• `eda_class_distribution.png` — distribusi fraud vs non-fraud <br>• `eda_amount_boxplot.png` — boxplot amount per kelas <br>• `eda_hour_distribution.png` — histogram jam transaksi per kelas <br>• `eda_correlation.png` — heatmap korelasi fitur <br>• `eval_confusion_matrix.png` — confusion matrix test set <br>• `eval_roc_curve.png` — kurva ROC + AUC <br>• `eval_pr_curve.png` — kurva Precision-Recall + PR-AUC <br>• `eval_training_history.png` — grafik loss & AUC per epoch |