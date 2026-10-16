"""
Módulo de Entrenamiento Formal con XGBoost (train_agent.py)
-----------------------------------------------------------
Carga los hiperparámetros optimizados desde CSV, entrena el modelo XGBoost 
empleando TimeSeriesSplit y guarda el modelo final listo para producción.
"""

import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

# Rutas absolutas del entorno
DATASET_PATH = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/master_ml_dataset.parquet"
PARAMS_CSV_PATH = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/best_hyperparameters.csv"
MODEL_OUTPUT_PATH = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/xgboost_trading_model.json"


def load_best_parameters(csv_path: str = PARAMS_CSV_PATH) -> dict:
    """Carga los hiperparámetros optimizados desde el archivo CSV."""
    if not os.path.exists(csv_path):
        print(f"⚠️ No se encontró el archivo de hiperparámetros en {csv_path}. Usando parámetros por defecto.")
        return {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "logloss"
        }
    
    df_params = pd.read_csv(csv_path)
    # Excluir columnas métricas auxiliares si existen
    params = df_params.drop(columns=["best_auc_roc"], errors="ignore").to_dict(orient="records")[0]
    # Asegurar tipos correctos para XGBoost
    params["n_estimators"] = int(params["n_estimators"])
    params["max_depth"] = int(params["max_depth"])
    params["random_state"] = int(params.get("random_state", 42))
    
    print("📋 Hiperparámetros óptimos cargados correctamente desde el CSV.")
    return params


def load_dataset(parquet_path: str = DATASET_PATH):
    """Carga el dataset maestro y separa las características del objetivo."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"❌ No se encontró el dataset maestro en: {parquet_path}")
        
    df = pd.read_parquet(parquet_path)
    if "date" in df.columns:
        df = df.sort_values("date")

    # Columnas que no deben ser usadas como predictoras
    ignore_cols = [
        "date", "ticker", "forward_max_high", 
        "forward_max_profit", "target_high_profit", "published_at", "source_url"
    ]
    features = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]

    X = df[features]
    target_col = "target_high_profit" if "target_high_profit" in df.columns else df.columns[-1]
    y = df[target_col].to_numpy()

    # Limpiar nulos
    valid_idx = ~np.isnan(y)
    X = X.loc[valid_idx]
    y = y[valid_idx]

    return features, X, y


def train_formal_model():
    print(f"🚀 Cargando dataset maestro desde: {DATASET_PATH}")
    features, X, y = load_dataset()
    print(f"📊 Dataset listo: {len(X):,} filas | {len(features)} características")

    # Cargar los hiperparámetros optimizados
    params = load_best_parameters()

    print("\n" + "="*60)
    print("📈 INICIANDO ENTRENAMIENTO CON VALIDACIÓN CRUZADA TEMPORAL")
    print("="*60)

    tscv = TimeSeriesSplit(n_splits=5)
    auc_scores = []

    fold = 1
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)

        preds_proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds_proba)
        auc_scores.append(score)
        
        print(f" Fold {fold} -> Train: {len(X_tr):6,} | Val: {len(X_val):6,} | AUC-ROC: {score:.4f}")
        fold += 1

    print("="*60)
    print(f"🏆 AUC-ROC PROMEDIO EN VALIDACIÓN CRUZADA: {np.mean(auc_scores):.4f}")
    print("="*60 + "\n")

    # Entrenamiento del modelo definitivo con todo el dataset histórico disponible
    print("💾 Entrenando modelo final de producción con todo el dataset...")
    final_model = XGBClassifier(**params)
    final_model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    final_model.save_model(MODEL_OUTPUT_PATH)
    print(f"🎯 Modelo formal guardado exitosamente en: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train_formal_model()
          
