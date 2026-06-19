"""
===========================================================
INFERENCE / DEPLOY - Deteksi Fraud dengan Model LSTM Tersimpan
===========================================================
Script ini untuk PREDIKSI menggunakan model yang sudah dilatih
sebelumnya (best_lstm_fraud_model.keras).

CARA PAKAI:
1. Upload 3 file ke folder yang sama dengan script ini:
   - best_lstm_fraud_model.keras
   - scaler.pkl
   - le_merchant.pkl
   (ketiganya dihasilkan otomatis oleh lstm-fraud-detection.py)
2. Jalankan script ini.
3. Lihat hasil prediksi fraud untuk data transaksi baru (simulasi).
===========================================================
"""

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

SEED = 42
np.random.seed(SEED)

# ===========================================================
# 0. KONFIGURASI -- HARUS SAMA PERSIS DENGAN SAAT TRAINING
# ===========================================================
MODEL_PATH = "best_lstm_fraud_model.keras"   # ganti sesuai lokasi file upload
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "le_merchant.pkl"
WINDOW_SIZE = 5
FEATURE_COLS = ["amount", "hour", "merchant_category_enc", "is_foreign", "time_since_last_tx"]


# ===========================================================
# 1. LOAD MODEL
# ===========================================================
print(">> Memuat model dari:", MODEL_PATH)
model = load_model(MODEL_PATH)
model.summary()


# ===========================================================
# 2. LOAD SCALER & ENCODER (langsung dari hasil training, tanpa rekonstruksi)
# ===========================================================
print("\n>> Memuat scaler dari:", SCALER_PATH)
scaler = joblib.load(SCALER_PATH)

print(">> Memuat encoder dari:", ENCODER_PATH)
le_merchant = joblib.load(ENCODER_PATH)

print(">> Scaler & LabelEncoder berhasil dimuat langsung dari hasil training.")
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
# CATATAN
# ===========================================================
# Script ini memuat scaler & encoder LANGSUNG dari file yang disimpan
# saat training (scaler.pkl, le_merchant.pkl) -- bukan direkonstruksi.
# Ini cara yang aman: scaler/encoder di sini PASTI identik dengan yang
# dipakai saat training, tidak ada risiko ketidaksesuaian.
#
# Pastikan ketiga file berikut berasal dari sesi training yang SAMA:
#   - best_lstm_fraud_model.keras
#   - scaler.pkl
#   - le_merchant.pkl
# Jika kamu retrain model dengan data baru, download ulang ketiganya.