"""
Módulo de Entrenamiento Final y Exportación de Artefactos (.joblib).
Entrena los modelos de Clasificación y Regresión con los mejores hiperparámetros
obtenidos por el agente optimizador y los guarda para el pipeline de inferencia/producción.
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
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def load_data(parquet_path: str = "data/parquet/ml_ready_dataset_enhanced.parquet"):
    """Carga y divide los datos de forma temporal."""
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


def train_and_save():
    os.makedirs("models", exist_ok=True)
    features, X_train, X_test, y_tr_cls, y_te_cls, y_tr_reg, y_te_reg = load_data()

    print(f"📦 Features de entrada ({len(features)}): {features}\n")

    # 1. ENTRENAR CLASIFICADOR CON HIPERPARÁMETROS OPTIMIZADOS
    print("🤖 Entrenando XGBoost Classifier Final...")
    best_params_cls = {
        "n_estimators": 256,
        "max_depth": 4,
        "learning_rate": 0.01192,
        "subsample": 0.6096,
        "colsample_bytree": 0.7025,
        "gamma": 2.4585,
        "random_state": 42,
        "eval_metric": "logloss"
    }

    clf = XGBClassifier(**best_params_cls)
    clf.fit(X_train, y_tr_cls)

    # Evaluación Clasificador
    preds_cls = clf.predict(X_test)
    probs_cls = clf.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_te_cls, probs_cls)
    cm = confusion_matrix(y_te_cls, preds_cls)

    print("\n--- 📊 REPORTE DE CLASIFICACIÓN (SWING TRADING) ---")
    print(f"AUC-ROC Score: {auc:.4f}")
    print("\nMatriz de Confusión:")
    print(f"  [TN: {cm[0][0]} | FP: {cm[0][1]}]")
    print(f"  [FN: {cm[1][0]} | TP: {cm[1][1]}]")
    print("\nMétricas detalladas:")
    print(classification_report(y_te_cls, preds_cls, target_names=["No Trigger (0)", "Profit >= 4% (1)"]))

    # Guardar modelo e hiperparámetros/features
    joblib.dump({
        "model": clf,
        "features": features,
        "metrics": {"auc_roc": auc}
    }, "models/xgboost_classifier.joblib")
    print("💾 Modelo guardado en: models/xgboost_classifier.joblib")

    print("\n" + "="*50 + "\n")

    # 2. ENTRENAR REGRESOR CON HIPERPARÁMETROS OPTIMIZADOS
    print("📈 Entrenando XGBoost Regressor Final...")
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

    # Evaluación Regresor
    preds_reg = reg.predict(X_test)
    mae = mean_absolute_error(y_te_reg, preds_reg)
    rmse = np.sqrt(mean_squared_error(y_te_reg, preds_reg))
    r2 = r2_score(y_te_reg, preds_reg)

    print("\n--- 📊 REPORTE DE REGRESIÓN (PROFIT ESPERADO) ---")
    print(f"MAE  (Error Medio Absoluto) : {mae:.4f} ({mae*100:.2f}%)")
    print(f"RMSE (Raíz Error Cuadrático) : {rmse:.4f} ({rmse*100:.2f}%)")
    print(f"R²   (Coeficiente Var.)     : {r2:.4f}")

    # Guardar modelo
    joblib.dump({
        "model": reg,
        "features": features,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2}
    }, "models/xgboost_regressor.joblib")
    print("💾 Modelo guardado en: models/xgboost_regressor.joblib")


if __name__ == "__main__":
    train_and_save()
  
