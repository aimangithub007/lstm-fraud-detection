"""
===========================================================
INFERENCE / DEPLOY - Deteksi Fraud dengan Model LSTM Tersimpan
===========================================================
Script ini untuk PREDIKSI menggunakan model yang sudah dilatih
sebelumnya (best_lstm_fraud_model.keras).

CARA PAKAI:
1. Upload file 'best_lstm_fraud_model.keras' ke folder yang sama
   dengan script ini (atau sesuaikan MODEL_PATH di bawah).
2. Jalankan script ini.
3. Lihat hasil prediksi fraud untuk data transaksi baru (simulasi).

PENTING:
- Scaler dan LabelEncoder di sini DIBUAT ULANG dari training data
  yang sama (karena hanya model .keras yang di-upload).
  Idealnya, scaler.pkl & le_merchant.pkl ikut disimpan saat training
  supaya konsisten 100% -- lihat catatan di bagian bawah script.
===========================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model

SEED = 42
np.random.seed(SEED)

# ===========================================================
# 0. KONFIGURASI -- HARUS SAMA PERSIS DENGAN SAAT TRAINING
# ===========================================================
MODEL_PATH = "best_lstm_fraud_model.keras"   # ganti sesuai lokasi file upload
WINDOW_SIZE = 5
FEATURE_COLS = ["amount", "hour", "merchant_category_enc", "is_foreign", "time_since_last_tx"]
MERCHANT_CATEGORIES = ["grocery", "electronics", "travel", "entertainment", "gas_station"]


# ===========================================================
# 1. LOAD MODEL
# ===========================================================
print(">> Memuat model dari:", MODEL_PATH)
model = load_model(MODEL_PATH)
model.summary()


# ===========================================================
# 2. REBUILD SCALER & ENCODER DARI DATA TRAINING YANG SAMA
# ===========================================================
# CATATAN PENTING:
# Scaler & encoder di sini dibuat ulang memakai fungsi generator
# yang SAMA dan SEED yang SAMA seperti saat training, supaya hasil
# fit-nya identik dengan training asli.
#
# Jika kamu mengubah fungsi generate data atau melatih ulang model
# dengan data berbeda, scaler/encoder ini HARUS diganti agar konsisten.
# Cara paling aman untuk produksi nyata: simpan scaler.pkl & le_merchant.pkl
# langsung saat training (lihat catatan di akhir file).

def generate_synthetic_transaction_data(n_users=500, max_tx_per_user=30,
                                         fraud_rate=0.04, seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    for user_id in range(n_users):
        n_tx = rng.integers(10, max_tx_per_user)
        base_amount = rng.normal(200_000, 50_000)
        for t in range(n_tx):
            is_fraud = 1 if rng.random() < fraud_rate else 0
            if is_fraud:
                amount = base_amount * rng.uniform(3, 8)
                hour = rng.choice([0, 1, 2, 3, 4, 23])
                is_foreign = rng.choice([0, 1], p=[0.3, 0.7])
                time_since_last = rng.uniform(0.01, 0.4)
                merchant = rng.choice(["electronics", "travel"])
            else:
                amount = max(10_000, rng.normal(base_amount, base_amount * 0.3))
                hour = rng.integers(6, 23)
                is_foreign = rng.choice([0, 1], p=[0.95, 0.05])
                time_since_last = rng.exponential(scale=12)
                merchant = rng.choice(MERCHANT_CATEGORIES)
            rows.append({
                "user_id": user_id, "tx_order": t, "amount": round(amount, 2),
                "hour": int(hour), "merchant_category": merchant,
                "is_foreign": int(is_foreign), "time_since_last_tx": round(time_since_last, 2),
                "is_fraud": is_fraud
            })
    return pd.DataFrame(rows)


df_train_ref = generate_synthetic_transaction_data()

le_merchant = LabelEncoder()
df_train_ref["merchant_category_enc"] = le_merchant.fit_transform(df_train_ref["merchant_category"])

scaler = StandardScaler()
scaler.fit(df_train_ref[FEATURE_COLS].values)

print("\n>> Scaler & LabelEncoder berhasil direkonstruksi dari data referensi training.")
print(">> Kelas merchant dikenal:", list(le_merchant.classes_))


# ===========================================================
# 3. BUAT DATA TEST (TRANSAKSI BARU UNTUK DIPREDIKSI)
# ===========================================================
# Kita buat beberapa skenario manual supaya hasilnya mudah diinterpretasi:
# - User A: riwayat transaksi NORMAL semua -> harapan: transaksi terakhir = non-fraud
# - User B: riwayat normal, lalu transaksi terakhir MENCURIGAKAN -> harapan: fraud terdeteksi
# - User C: baru (kurang dari WINDOW_SIZE transaksi) -> uji padding

test_transactions = pd.DataFrame([
    # --- User A: pola normal semua ---
    {"user_id": "A", "tx_order": 0, "amount": 180000, "hour": 14, "merchant_category": "grocery",
     "is_foreign": 0, "time_since_last_tx": 10.0},
    {"user_id": "A", "tx_order": 1, "amount": 195000, "hour": 16, "merchant_category": "gas_station",
     "is_foreign": 0, "time_since_last_tx": 8.0},
    {"user_id": "A", "tx_order": 2, "amount": 175000, "hour": 19, "merchant_category": "entertainment",
     "is_foreign": 0, "time_since_last_tx": 6.5},
    {"user_id": "A", "tx_order": 3, "amount": 205000, "hour": 12, "merchant_category": "grocery",
     "is_foreign": 0, "time_since_last_tx": 9.0},
    {"user_id": "A", "tx_order": 4, "amount": 190000, "hour": 15, "merchant_category": "electronics",
     "is_foreign": 0, "time_since_last_tx": 7.0},

    # --- User B: normal, lalu transaksi TERAKHIR mencurigakan ---
    {"user_id": "B", "tx_order": 0, "amount": 150000, "hour": 13, "merchant_category": "grocery",
     "is_foreign": 0, "time_since_last_tx": 11.0},
    {"user_id": "B", "tx_order": 1, "amount": 160000, "hour": 17, "merchant_category": "gas_station",
     "is_foreign": 0, "time_since_last_tx": 9.0},
    {"user_id": "B", "tx_order": 2, "amount": 145000, "hour": 20, "merchant_category": "entertainment",
     "is_foreign": 0, "time_since_last_tx": 5.0},
    {"user_id": "B", "tx_order": 3, "amount": 155000, "hour": 14, "merchant_category": "grocery",
     "is_foreign": 0, "time_since_last_tx": 10.0},
    {"user_id": "B", "tx_order": 4, "amount": 980000, "hour": 2, "merchant_category": "electronics",
     "is_foreign": 1, "time_since_last_tx": 0.15},  # <- mencurigakan: amount tinggi, jam dini hari, luar negeri, cepat

    # --- User C: transaksi baru, riwayat sangat sedikit (uji padding) ---
    {"user_id": "C", "tx_order": 0, "amount": 700000, "hour": 1, "merchant_category": "travel",
     "is_foreign": 1, "time_since_last_tx": 0.05},
])

print("\n=== DATA TEST (TRANSAKSI BARU) ===")
print(test_transactions)


# ===========================================================
# 4. PREPROCESSING DATA TEST (samakan langkahnya dengan training)
# ===========================================================
# Encode merchant_category. Jika ada kategori baru yang tidak dikenal
# encoder, fallback ke kategori paling umum supaya tidak error.
known_categories = set(le_merchant.classes_)

def safe_encode_merchant(cat):
    if cat not in known_categories:
        print(f"   [WARNING] Kategori '{cat}' tidak dikenal saat training, fallback ke 'grocery'")
        cat = "grocery"
    return le_merchant.transform([cat])[0]

test_transactions["merchant_category_enc"] = test_transactions["merchant_category"].apply(safe_encode_merchant)


# ===========================================================
# 5. BUILD SEQUENCE UNTUK TRANSAKSI TERAKHIR SETIAP USER
# ===========================================================
def build_sequence_for_prediction(data, feature_cols, user_col="user_id",
                                   order_col="tx_order", window_size=WINDOW_SIZE):
    """
    Sama seperti build_sequences saat training, tapi tanpa label
    (karena ini data yang AKAN diprediksi).
    Mengembalikan sequence untuk TRANSAKSI TERAKHIR per user saja.
    """
    sequences = []
    user_ids = []

    data_sorted = data.sort_values([user_col, order_col])

    for user_id, group in data_sorted.groupby(user_col):
        feats = group[feature_cols].values
        n = len(feats)

        # Ambil window untuk transaksi TERAKHIR user ini
        window = feats[max(0, n - window_size):n]

        if len(window) < window_size:
            pad = np.zeros((window_size - len(window), len(feature_cols)))
            window = np.vstack([pad, window])

        sequences.append(window)
        user_ids.append(user_id)

    return np.array(sequences), user_ids


X_new, user_ids = build_sequence_for_prediction(test_transactions, FEATURE_COLS)
print("\nShape sequence data test:", X_new.shape)


# ===========================================================
# 6. SCALING (pakai scaler yang sudah di-fit dari training)
# ===========================================================
n_features = X_new.shape[2]
X_new_2d = X_new.reshape(-1, n_features)
X_new_scaled = scaler.transform(X_new_2d).reshape(X_new.shape)


# ===========================================================
# 7. PREDIKSI
# ===========================================================
fraud_probabilities = model.predict(X_new_scaled).ravel()
THRESHOLD = 0.5
fraud_predictions = (fraud_probabilities >= THRESHOLD).astype(int)


# ===========================================================
# 8. TAMPILKAN HASIL
# ===========================================================
results = pd.DataFrame({
    "user_id": user_ids,
    "fraud_probability": fraud_probabilities,
    "prediction": ["FRAUD" if p == 1 else "Non-Fraud" for p in fraud_predictions]
})

print("\n=== HASIL DETEKSI FRAUD (transaksi terakhir per user) ===")
print(results.to_string(index=False))

print("\n>> Interpretasi cepat:")
for _, row in results.iterrows():
    flag = "🚨" if row["prediction"] == "FRAUD" else "✅"
    print(f"   {flag} User {row['user_id']}: probabilitas fraud = {row['fraud_probability']:.4f} -> {row['prediction']}")


# ===========================================================
# CATATAN UNTUK PRODUKSI YANG LEBIH AMAN
# ===========================================================
# Script ini merekonstruksi scaler & encoder dari ulang generate data
# training (karena hanya file .keras yang di-upload). Ini AMAN selama
# fungsi generator & seed tidak berubah dari training asli.
#
# Untuk deployment produksi sesungguhnya, sebaiknya saat training kamu
# juga simpan scaler & encoder langsung, contoh:
#
#   import joblib
#   joblib.dump(scaler, "scaler.pkl")
#   joblib.dump(le_merchant, "le_merchant.pkl")
#
# lalu saat inference, load langsung tanpa rekonstruksi:
#
#   scaler = joblib.load("scaler.pkl")
#   le_merchant = joblib.load("le_merchant.pkl")
#
# Ini menghindari risiko ketidaksesuaian jika generator data berubah.