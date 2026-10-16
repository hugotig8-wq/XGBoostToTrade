"""
Módulo de Entrenamiento y Comparación Multi-Horizonte con XGBRegressor (train_agent.py)
------------------------------------------------------------------------------------
Carga los hiperparámetros optimizados, entrena modelos independientes para cada 
horizonte de retorno (1h, 3h, 6h, 12h, 1d) empleando TimeSeriesSplit, 
compara sus resultados de RMSE y guarda los modelos listos para producción.
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
MODEL_OUTPUT_DIR = "/workspaces/XGBoostToTrade/CollectorAgent/data/models/"


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
    params = df_params.drop(columns=["best_rmse"], errors="ignore").to_dict(orient="records")[0]
    
    params["n_estimators"] = int(params["n_estimators"])
    params["max_depth"] = int(params["max_depth"])
    params["random_state"] = int(params.get("random_state", 42))
    params["eval_metric"] = "rmse"
    
    return params


def load_dataset(parquet_path: str = DATASET_PATH):
    """Carga el dataset maestro y separa las características."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"❌ No se encontró el dataset maestro en: {parquet_path}")
        
    df = pd.read_parquet(parquet_path)
    if "date" in df.columns:
        df = df.sort_values("date")

    # Columnas estrictamente prohibidas para evitar Data Leakage
    ignore_cols = [
        "date", "ticker", "published_at", "source_url",
        "forward_max_high", "forward_max_profit", "target_high_profit",
        "return_1h", "return_3h", "return_6h", "return_12h", "return_1d"
    ]
    
    numeric_cols = df.select_dtypes(include=[np.number, "float", "int", "bool"]).columns.tolist()
    features = [c for c in numeric_cols if c not in ignore_cols]

    X = df[features]
    return df, X, features


def train_and_compare_multi_horizon():
    print(f"🚀 Cargando dataset maestro desde: {DATASET_PATH}")
    df, X, features = load_dataset()
    print(f"📊 Dataset base: {len(X):,} filas | {len(features)} características válidas (sin leakage)")

    params = load_best_parameters()
    
    # Horizontes temporales a evaluar y comparar
    horizons = ["return_1h", "return_3h", "return_6h", "return_12h", "return_1d"]
    available_horizons = [h for h in horizons if h in df.columns]

    if not available_horizons:
        print("❌ No se encontraron columnas de retorno en el dataset.")
        return

    tscv = TimeSeriesSplit(n_splits=5)
    comparison_results = {}

    print("\n" + "="*70)
    print("📈 INICIANDO ENTRENAMIENTO Y COMPARACIÓN MULTI-HORIZONTE (TimeSeriesSplit)")
    print("="*70)

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    for target_col in available_horizons:
        print(f"\n🎯 Evaluando Horizonte Objetivo: [{target_col}]")
        
        # Preparar target y limpiar nulos específicos de este horizonte
        y = pd.to_numeric(df[target_col], errors="coerce").to_numpy()
        valid_idx = ~pd.isna(y)
        X_target = X.loc[valid_idx]
        y_target = y[valid_idx]

        fold = 1
        rmse_scores = []

        for train_idx, val_idx in tscv.split(X_target):
            X_tr, X_val = X_target.iloc[train_idx], X_target.iloc[val_idx]
            y_tr, y_val = y_target[train_idx], y_target[val_idx]

            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr)

            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            rmse_scores.append(rmse)
            fold += 1

        mean_rmse = np.mean(rmse_scores)
        comparison_results[target_col] = mean_rmse
        print(f"   -> RMSE Promedio para {target_col}: {mean_rmse:.5f}")

        # Entrenar modelo final de producción para este horizonte con todo el histórico disponible
        final_model = XGBRegressor(**params)
        final_model.fit(X_target, y_target)
        
        model_path = os.path.join(MODEL_OUTPUT_DIR, f"xgboost_model_{target_col}.json")
        final_model.save_model(model_path)
        print(f"   💾 Modelo guardado en: {model_path}")

    # Tabla Comparativa Final
    print("\n" + "="*70)
    print("🏆 TABLA COMPARATIVA DE RENDIMIENTO (RMSE por Horizonte)")
    print("="*70)
    sorted_comparison = sorted(comparison_results.items(), key=lambda x: x[1])
    for horizon, rmse in sorted_comparison:
        print(f"  * {horizon:10} | RMSE Promedio: {rmse:.5f}")
    print("="*70)
    print("✅ ¡Entrenamiento multi-horizonte y comparativa completados con éxito!\n")


if __name__ == "__main__":
    train_and_compare_multi_horizon()
        
