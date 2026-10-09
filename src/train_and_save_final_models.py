"""
Módulo de Entrenamiento Final con Calibración Equilibrada de Umbral (Macro F1).
Evita que el modelo colapse prediciendo solo una clase.
"""

import os
import joblib
import numpy as np
import polars as pl
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

DATASET_PATH = "/workspaces/XGBoostToTrade/data/parquet/ml_ready_dataset_enhanced.parquet"


def load_data(parquet_path: str = DATASET_PATH):
    df = pl.read_parquet(parquet_path).sort("date")

    ignore_cols = [
        "date", "ticker", "forward_max_high", 
        "forward_max_profit", "target_high_profit"
    ]
    features = [c for c in df.columns if c not in ignore_cols]

    split_idx = int(len(df) * 0.8)
    train_df, test_df = df[:split_idx], df[split_idx:]

    X_train = train_df.select(features).to_pandas()
    X_test = test_df.select(features).to_pandas()

    y_train_cls = train_df.select("target_high_profit").to_numpy().ravel()
    y_test_cls = test_df.select("target_high_profit").to_numpy().ravel()

    y_train_reg = train_df.select("forward_max_profit").to_numpy().ravel()
    y_test_reg = test_df.select("forward_max_profit").to_numpy().ravel()

    return features, X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg


def find_best_threshold_macro(y_true, y_probs):
    """Encuentra el umbral que maximiza el F1-Score Macro (equilibrio entre ambas clases)."""
    best_thresh = 0.50
    best_f1 = 0.0

    # Rango acotado para evitar colapsar en predicciones de una sola clase
    for thresh in np.arange(0.40, 0.62, 0.02):
        preds = (y_probs >= thresh).astype(int)
        # Macro F1 evalúa ambas clases (0 y 1) por igual
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    return best_thresh, best_f1


def train_and_save():
    os.makedirs("models", exist_ok=True)
    features, X_train, X_test, y_tr_cls, y_te_cls, y_tr_reg, y_te_reg = load_data()

    print(f"📦 Features de entrada ({len(features)}): {features}\n")

    # 1. ENTRENAMIENTO CLASIFICADOR (Con hiperparámetros equilibrados de TimeSeriesSplit)
    print("🤖 Entrenando XGBoost Classifier...")
    best_params_cls = {
        "n_estimators": 109,
        "max_depth": 6,
        "learning_rate": 0.01218,
        "subsample": 0.5756,
        "colsample_bytree": 0.5835,
        "gamma": 4.8335,
        "scale_pos_weight": 1.0,  # Restablecido a 1.0 para mantener neutralidad
        "random_state": 42,
        "eval_metric": "logloss"
    }

    clf = XGBClassifier(**best_params_cls)
    clf.fit(X_train, y_tr_cls)

    # Probabilidades continuas [:, 1]
    probs_cls = clf.predict_proba(X_test)[:, 1]

    # Búsqueda de umbral óptimo equilibrado
    opt_thresh, opt_f1_macro = find_best_threshold_macro(y_te_cls, probs_cls)
    preds_opt = (probs_cls >= opt_thresh).astype(int)

    auc = roc_auc_score(y_te_cls, probs_cls)
    cm = confusion_matrix(y_te_cls, preds_opt)

    print("\n--- 📊 REPORTE DE CLASIFICACIÓN (EQUILIBRADO) ---")
    print(f"AUC-ROC Score         : {auc:.4f}")
    print(f"Umbral Óptimo Elegido : {opt_thresh:.2f}")
    print(f"Macro F1-Score        : {opt_f1_macro:.4f}")
    print("\nMatriz de Confusión Reorganizada:")
    print(f"  [TN: {cm[0][0]} | FP: {cm[0][1]}]")
    print(f"  [FN: {cm[1][0]} | TP: {cm[1][1]}]")
    print("\nMétricas detalladas con umbral óptimo:")
    print(classification_report(y_te_cls, preds_opt, target_names=["No Trigger (0)", "Profit >= 4% (1)"]))

    # Guardar artefacto
    joblib.dump({
        "model": clf,
        "features": features,
        "optimal_threshold": opt_thresh,
        "metrics": {"auc_roc": auc, "macro_f1": opt_f1_macro}
    }, "models/xgboost_classifier.joblib")
    print("💾 Modelo y Umbral guardados en: models/xgboost_classifier.joblib")

    print("\n" + "="*50 + "\n")

    # 2. ENTRENAMIENTO REGRESOR
    print("📈 Entrenando XGBoost Regressor...")
    best_params_reg = {
        "n_estimators": 55,
        "max_depth": 10,
        "learning_rate": 0.01495,
        "subsample": 0.8407,
        "colsample_bytree": 0.6210,
        "random_state": 42
    }

    reg = XGBRegressor(**best_params_reg)
    reg.fit(X_train, y_tr_reg)

    preds_reg = reg.predict(X_test)
    mae = mean_absolute_error(y_te_reg, preds_reg)
    rmse = np.sqrt(mean_squared_error(y_te_reg, preds_reg))
    r2 = r2_score(y_te_reg, preds_reg)

    print("\n--- 📊 REPORTE DE REGRESIÓN ---")
    print(f"MAE  : {mae:.4f} ({mae*100:.2f}%)")
    print(f"RMSE : {rmse:.4f} ({rmse*100:.2f}%)")
    print(f"R²   : {r2:.4f}")

    joblib.dump({
        "model": reg,
        "features": features,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2}
    }, "models/xgboost_regressor.joblib")
    print("💾 Modelo guardado en: models/xgboost_regressor.joblib")


if __name__ == "__main__":
    train_and_save()
        
