"""
===========================================================
LSTM untuk Fraud Detection - Sequential Transaction per User
===========================================================
Kasus: Setiap user punya riwayat transaksi berurutan (time series).
Tujuan: Prediksi apakah TRANSAKSI TERAKHIR dalam sequence adalah FRAUD.

Alur lengkap:
1. EDA
2. Preprocessing (scaling, encoding)
3. Sequence Building (per user, sliding window)
4. Train-test split (berbasis user, bukan acak antar baris!)
5. Build LSTM model
6. Training + Tuning (callbacks, hyperparameter)
7. Evaluasi (confusion matrix, classification report, ROC-AUC)
===========================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ===========================================================
# 1. SIMULASI DATASET (ganti dengan data asli: pd.read_csv(...))
# ===========================================================
def generate_synthetic_transaction_data(n_users=500, max_tx_per_user=30,
                                         fraud_rate=0.04, seed=SEED):
    """
    Setiap user punya jumlah transaksi yang berbeda-beda.
    Fitur per transaksi: amount, hour, merchant_category, is_foreign, time_since_last_tx
    Label: is_fraud (0/1).

    PENDEKATAN: fraud_rate menentukan proporsi transaksi yang DITANDAI fraud
    secara eksplisit (bukan ditebak dari kombinasi kondisi yang jarang terjadi).
    Setelah ditandai, baru fitur-fiturnya "dibuat" mencurigakan -- supaya
    proporsi fraud terkontrol dan model punya cukup contoh untuk belajar.
    """
    rng = np.random.default_rng(seed)
    rows = []

    merchant_categories = ["grocery", "electronics", "travel", "entertainment", "gas_station"]

    for user_id in range(n_users):
        n_tx = rng.integers(10, max_tx_per_user)
        base_amount = rng.normal(200_000, 50_000)  # rata-rata transaksi user ini (IDR)

        for t in range(n_tx):
            # Tentukan dulu apakah transaksi ini fraud (rate terkontrol)
            is_fraud = 1 if rng.random() < fraud_rate else 0

            if is_fraud:
                # Transaksi fraud: pola mencurigakan disuntikkan langsung
                amount = base_amount * rng.uniform(3, 8)          # amount melonjak tiba-tiba
                hour = rng.choice([0, 1, 2, 3, 4, 23])             # jam tidak wajar (dini hari)
                is_foreign = rng.choice([0, 1], p=[0.3, 0.7])      # cenderung transaksi luar negeri
                time_since_last = rng.uniform(0.01, 0.4)           # transaksi sangat cepat berurutan
                merchant = rng.choice(["electronics", "travel"])  # merchant favorit fraud
            else:
                # Transaksi normal
                amount = max(10_000, rng.normal(base_amount, base_amount * 0.3))
                hour = rng.integers(6, 23)
                is_foreign = rng.choice([0, 1], p=[0.95, 0.05])
                time_since_last = rng.exponential(scale=12)
                merchant = rng.choice(merchant_categories)

            rows.append({
                "user_id": user_id,
                "tx_order": t,
                "amount": round(amount, 2),
                "hour": int(hour),
                "merchant_category": merchant,
                "is_foreign": int(is_foreign),
                "time_since_last_tx": round(time_since_last, 2),
                "is_fraud": is_fraud
            })

    return pd.DataFrame(rows)


df = generate_synthetic_transaction_data()
print("Shape dataset:", df.shape)
print(df.head(10))


# ===========================================================
# 2. EDA (Exploratory Data Analysis)
# ===========================================================
print("\n=== INFO DATASET ===")
print(df.info())

print("\n=== DISTRIBUSI LABEL (CLASS IMBALANCE CHECK) ===")
print(df["is_fraud"].value_counts())
print(df["is_fraud"].value_counts(normalize=True))

# Visualisasi distribusi kelas
plt.figure(figsize=(5, 4))
sns.countplot(x="is_fraud", data=df)
plt.title("Distribusi Fraud vs Non-Fraud")
plt.savefig("eda_class_distribution.png", dpi=100, bbox_inches="tight")
plt.close()

# Distribusi amount berdasarkan label
plt.figure(figsize=(7, 4))
sns.boxplot(x="is_fraud", y="amount", data=df)
plt.title("Distribusi Amount: Fraud vs Non-Fraud")
plt.savefig("eda_amount_boxplot.png", dpi=100, bbox_inches="tight")
plt.close()

# Distribusi jam transaksi
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="hour", hue="is_fraud", bins=24, multiple="stack")
plt.title("Distribusi Jam Transaksi: Fraud vs Non-Fraud")
plt.savefig("eda_hour_distribution.png", dpi=100, bbox_inches="tight")
plt.close()

# Korelasi fitur numerik
plt.figure(figsize=(6, 5))
numeric_cols = ["amount", "hour", "is_foreign", "time_since_last_tx", "is_fraud"]
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korelasi Fitur Numerik")
plt.savefig("eda_correlation.png", dpi=100, bbox_inches="tight")
plt.close()

print("\n>> Catatan EDA: cek class imbalance -- jika fraud << non-fraud, ")
print(">> perlu class_weight saat training (lihat bagian Training).")


# ===========================================================
# 3. PREPROCESSING
# ===========================================================
# Encoding kategorikal
le_merchant = LabelEncoder()
df["merchant_category_enc"] = le_merchant.fit_transform(df["merchant_category"])

feature_cols = ["amount", "hour", "merchant_category_enc", "is_foreign", "time_since_last_tx"]

# PENTING: scaling fit HANYA di data training nanti (hindari data leakage).
# Di sini kita siapkan dulu, scaler.fit_transform dipanggil setelah split.


# ===========================================================
# 4. SEQUENCE BUILDING (sliding window per user)
# ===========================================================
def build_sequences(data, feature_cols, label_col="is_fraud",
                     user_col="user_id", order_col="tx_order",
                     window_size=5):
    """
    Untuk setiap user, buat sequence sepanjang `window_size` transaksi berurutan.
    Label sequence = label transaksi TERAKHIR dalam window (yang ingin diprediksi).

    Jika user punya transaksi < window_size, di-pad dengan 0 di awal (pre-padding).
    """
    sequences = []
    labels = []

    data_sorted = data.sort_values([user_col, order_col])

    for user_id, group in data_sorted.groupby(user_col):
        feats = group[feature_cols].values
        labs = group[label_col].values
        n = len(feats)

        for i in range(n):
            start = max(0, i - window_size + 1)
            window = feats[start:i + 1]

            # Pre-padding jika window belum penuh
            if len(window) < window_size:
                pad = np.zeros((window_size - len(window), len(feature_cols)))
                window = np.vstack([pad, window])

            sequences.append(window)
            labels.append(labs[i])

    return np.array(sequences), np.array(labels)


WINDOW_SIZE = 5
X_seq, y_seq = build_sequences(df, feature_cols, window_size=WINDOW_SIZE)
print("\nShape sequence X:", X_seq.shape)  # (n_samples, window_size, n_features)
print("Shape label y:", y_seq.shape)


# ===========================================================
# 5. TRAIN-TEST SPLIT
# ===========================================================
# Split langsung di level sequence (sudah representasi 1 transaksi + history-nya).
# Stratify supaya proporsi fraud tetap terjaga di train & test.
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=SEED, stratify=y_seq
)

print("\nTrain shape:", X_train.shape, " Fraud ratio train:", y_train.mean())
print("Test shape :", X_test.shape, " Fraud ratio test :", y_test.mean())

# ===========================================================
# 6. SCALING (fit HANYA di train, transform ke train & test)
# ===========================================================
n_features = X_train.shape[2]
scaler = StandardScaler()

# Reshape ke 2D untuk scaler, lalu kembalikan ke 3D
X_train_2d = X_train.reshape(-1, n_features)
X_test_2d = X_test.reshape(-1, n_features)

scaler.fit(X_train_2d)  # fit HANYA di training -> hindari data leakage

X_train_scaled = scaler.transform(X_train_2d).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape)


# ===========================================================
# 7. HANDLE CLASS IMBALANCE
# ===========================================================
class_weights_arr = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weight_dict = {0: class_weights_arr[0], 1: class_weights_arr[1]}
print("\nClass weights:", class_weight_dict)


# ===========================================================
# 8. BUILD MODEL LSTM
# ===========================================================
def build_lstm_model(window_size, n_features, lstm_units=64, dropout_rate=0.3, learning_rate=1e-3):
    model = Sequential([
        Input(shape=(window_size, n_features)),

        LSTM(lstm_units, return_sequences=True),
        BatchNormalization(),
        Dropout(dropout_rate),

        LSTM(lstm_units // 2),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(32, activation="relu"),
        Dropout(dropout_rate / 2),

        Dense(1, activation="sigmoid")  # output probabilitas fraud
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")]
    )
    return model


model = build_lstm_model(WINDOW_SIZE, n_features)
model.summary()


# ===========================================================
# 9. CALLBACKS (bagian dari "tuning" proses training)
# ===========================================================
callbacks = [
    EarlyStopping(monitor="val_auc", mode="max", patience=10,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                       min_lr=1e-6, verbose=1),
    ModelCheckpoint("best_lstm_fraud_model.keras", monitor="val_auc",
                     mode="max", save_best_only=True, verbose=0)
]


# ===========================================================
# 10. TRAINING
# ===========================================================
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)


# ===========================================================
# 11. HYPERPARAMETER TUNING (manual grid search sederhana)
# ===========================================================
def tune_hyperparameters(X_train, y_train, X_val, y_val, window_size, n_features):
    """
    Grid search sederhana atas beberapa kombinasi hyperparameter.
    Untuk dataset besar / produksi, gunakan Keras Tuner atau Optuna.
    """
    param_grid = [
        {"lstm_units": 32, "dropout_rate": 0.2, "learning_rate": 1e-3},
        {"lstm_units": 64, "dropout_rate": 0.3, "learning_rate": 1e-3},
        {"lstm_units": 64, "dropout_rate": 0.4, "learning_rate": 5e-4},
    ]

    best_auc = -1
    best_params = None
    best_model = None

    for params in param_grid:
        print(f"\n>> Mencoba kombinasi: {params}")
        m = build_lstm_model(window_size, n_features, **params)
        es = EarlyStopping(monitor="val_auc", mode="max", patience=5,
                            restore_best_weights=True, verbose=0)
        m.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=30, batch_size=32, callbacks=[es], verbose=0)

        val_auc = max(m.history.history["val_auc"])
        print(f"   val_auc terbaik: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_params = params
            best_model = m

    print(f"\n=== Hyperparameter terbaik: {best_params} (val_auc={best_auc:.4f}) ===")
    return best_model, best_params


# Contoh pemanggilan tuning (opsional, memakan waktu lebih lama):
# X_tr2, X_val2, y_tr2, y_val2 = train_test_split(
#     X_train_scaled, y_train, test_size=0.2, random_state=SEED, stratify=y_train
# )
# best_model, best_params = tune_hyperparameters(
#     X_tr2, y_tr2, X_val2, y_val2, WINDOW_SIZE, n_features
# )


# ===========================================================
# 12. EVALUASI MODEL FINAL
# ===========================================================
y_pred_prob = model.predict(X_test_scaled).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=["Non-Fraud", "Fraud"]))

print("\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix")
plt.savefig("eval_confusion_matrix.png", dpi=100, bbox_inches="tight")
plt.close()

auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"\nROC-AUC Score: {auc_score:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("eval_roc_curve.png", dpi=100, bbox_inches="tight")
plt.close()

# Precision-Recall Curve -- LEBIH RELEVAN untuk data imbalance seperti fraud detection
# (ROC-AUC bisa terlihat "bagus" walau model jelek kalau kelas minoritas sangat sedikit)
from sklearn.metrics import average_precision_score
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = average_precision_score(y_test, y_pred_prob)
print(f"PR-AUC Score: {pr_auc:.4f}  (baseline acak = rasio fraud di test = {y_test.mean():.4f})")

plt.figure(figsize=(5, 5))
plt.plot(recall_vals, precision_vals, label=f"PR-AUC = {pr_auc:.3f}")
plt.axhline(y_test.mean(), linestyle="--", color="gray", label="Baseline (tebak acak)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig("eval_pr_curve.png", dpi=100, bbox_inches="tight")
plt.close()

# Training history plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["loss"], label="train_loss")
axes[0].plot(history.history["val_loss"], label="val_loss")
axes[0].set_title("Loss per Epoch")
axes[0].legend()

axes[1].plot(history.history["auc"], label="train_auc")
axes[1].plot(history.history["val_auc"], label="val_auc")
axes[1].set_title("AUC per Epoch")
axes[1].legend()

plt.savefig("eval_training_history.png", dpi=100, bbox_inches="tight")
plt.close()

print("\n>> Selesai. Model terbaik tersimpan di 'best_lstm_fraud_model.keras'")
print(">> Semua plot EDA & evaluasi tersimpan sebagai file .png")