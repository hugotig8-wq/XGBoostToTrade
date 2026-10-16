"""
Agente Optimizador de Hiperparámetros con TimeSeriesSplit y Regresión XGBoost.
Evalúa XGBoost mediante validación cruzada temporal y exporta los mejores parámetros.
"""

import os
import optuna
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# Silenciar mensajes informativos de Optuna para mantener la consola limpia
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Rutas absolutas actualizadas al entorno de trabajo
DATASET_PATH = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/master_ml_dataset.parquet"
OUTPUT_CSV_PATH = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/best_hyperparameters.csv"


def load_dataset(parquet_path: str = DATASET_PATH):
    """Carga el dataset maestro ordenado por fecha y separa las características."""
    if not os.path.exists(parquet_path):
        print(f"⚠️ Advertencia: No se encontró el archivo en {parquet_path}")
        
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
    
    # Seleccionar la columna objetivo (target) de retorno continuo (ej. return_1h)
    target_col = "return_1h" if "return_1h" in df.columns else features[0]
    
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy()

    # Limpieza de nulos
    valid_idx = ~pd.isna(y)
    X = X.loc[valid_idx]
    y = y[valid_idx]

    return features, X, y


def objective_cv(trial, X, y):
    """Función objetivo que evalúa hiperparámetros con TimeSeriesSplit y Regresión."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 350),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "gamma": trial.suggest_float("gamma", 0.5, 5.0),
        "random_state": 42,
        "eval_metric": "rmse"
    }

    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)


def main():
    print(f"🚀 Cargando dataset desde: {DATASET_PATH}")
    features, X, y = load_dataset()
    print(f"📊 Dataset cargado: {len(X)} filas | {len(features)} características")

    print("\n🔎 AGENTE OPTIMIZANDO REGRESIÓN CON TIME-SERIES CROSS-VALIDATION (5 Folds)...")
    # Minimizamos el RMSE, por lo que direction="minimize"
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective_cv(trial, X, y), n_trials=30)

    print("\n" + "="*55)
    print(f"🏆 MEJOR RMSE PROMEDIO EN VALIDACIÓN CRUZADA: {study.best_value:.5f}")
    print("\nMejores Hiperparámetros Encontrados:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*55)

    # Guardar los mejores hiperparámetros en formato CSV
    best_params_df = pd.DataFrame([study.best_params])
    best_params_df["best_rmse"] = study.best_value
    
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    best_params_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"💾 Mejores hiperparámetros guardados exitosamente en: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
    
