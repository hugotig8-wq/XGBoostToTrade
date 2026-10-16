"""
Módulo de Entrenamiento Formal con XGBRegressor (train_agent.py)
--------------------------------------------------------------
Carga los hiperparámetros optimizados desde CSV, entrena el modelo XGBoost 
empleando TimeSeriesSplit y guarda el modelo final de regresión listo para producción.
"""

import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# Rutas absolutas del entorno
DATASET_PATH = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/master_ml_dataset.parquet"
PARAMS_CSV_PATH = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/best_hyperparameters.csv"
MODEL_OUTPUT_PATH = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/xgboost_trading_model.json"


def load_best_parameters(csv_path: str = PARAMS_CSV_PATH) -> dict:
    """Carga los hiperparámetros optimizados desde el archivo CSV."""
    if not os.path.exists(csv_path):
        print(f"⚠️ No se encontró el archivo de hiperparámetros en {csv_path}. Usando parámetros por defecto.")
        return {
            "n_estimators": 250,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": "rmse"
        }
    
    df_params = pd.read_csv(csv_path)
    # Excluir columnas métricas auxiliares si existen
    params = df_params.drop(columns=["best_rmse"], errors="ignore").to_dict(orient="records")[0]
    
    # Asegurar tipos correctos para XGBoost
    params["n_estimators"] = int(params["n_estimators"])
    params["max_depth"] = int(params["max_depth"])
    params["random_state"] = int(params.get("random_state", 42))
    params["eval_metric"] = "rmse"
    
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
        "forward_max_profit", "target_high_profit", "published_at", "source_url",
        "return_1h", "return_3h", "return_6h", "return_12h", "return_1d"
    ]
    
    # Filtrar solo columnas estrictamente numéricas
    numeric_cols = df.select_dtypes(include=[np.number, "float", "int", "bool"]).columns.tolist()
    features = [c for c in numeric_cols if c not in ignore_cols]

    X = df[features]
    
    # Seleccionar la columna objetivo (target) de retorno continuo
    target_col = "return_1h" if "return_1h" in df.columns else features[0]
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy()

    # Limpieza robusta de nulos con pd.isna()
    valid_idx = ~pd.isna(y)
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
    print("📈 INICIANDO ENTRENAMIENTO CON VALIDACIÓN CRUZADA TEMPORAL (RMSE)")
    print("="*60)

    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores = []

    fold = 1
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)
        
        print(f" Fold {fold} -> Train: {len(X_tr):6,} | Val: {len(X_val):6,} | RMSE: {rmse:.5f}")
        fold += 1

    print("="*60)
    print(f"🏆 RMSE PROMEDIO EN VALIDACIÓN CRUZADA: {np.mean(rmse_scores):.5f}")
    print("="*60 + "\n")

    # Entrenamiento del modelo definitivo con todo el dataset histórico disponible
    print("💾 Entrenando modelo final de producción con todo el dataset...")
    final_model = XGBRegressor(**params)
    final_model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    final_model.save_model(MODEL_OUTPUT_PATH)
    print(f"🎯 Modelo formal de regresión guardado exitosamente en: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train_formal_model()
    
