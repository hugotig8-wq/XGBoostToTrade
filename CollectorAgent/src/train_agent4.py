import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

DATASET_PATH = "/home/humberto/master_ml_dataset.parquet"
MODEL_OUTPUT_DIR = "/home/humberto/"

def load_best_parameters() -> dict:
    params = {
        "n_estimators": 250,
        "max_depth": 5,
        "gamma": 0,  # <--- CRUCIAL: 0.01 evitaba que el modelo aprendiera
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "rmse"
    }
    return params

def load_dataset(parquet_path: str = DATASET_PATH):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"❌ No se encontró el dataset maestro en: {parquet_path}")
        
    df = pd.read_parquet(parquet_path)
    # Si el dataset quedó ordenado por fecha de noticia, lo ordenamos por fecha de vela
    if "datetime" in df.columns:
        df = df.sort_values("datetime")

    # EXCLUSIÓN CRUCIAL: price_before_news es un leak directo del target
    ignore_cols = [
        "date", "datetime", "ticker", "published_at", "published_at_dt", "source_url",
        "forward_max_high", "forward_max_profit", "target_high_profit",
        "return_1h", "return_3h", "return_6h", "return_12h", "return_1d",
        "return_2d", "return_3d", "return_5d", "return_10d",
        "source", "event_actor", "subject_entity", "sector", 
        "event_type", "relationship_reason", "return_status", "price_timestamp",
        "price_before_news" # <--- LEAK ELIMINADO
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
    horizons = ["return_1h", "return_3h", "return_6h", "return_12h", "return_1d"]
    available_horizons = [h for h in horizons if h in df.columns]

    if not available_horizons:
        print("❌ No se encontraron columnas de retorno en el dataset.")
        return

    tscv = TimeSeriesSplit(n_splits=2)
    comparison_results = {}

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    for target_col in available_horizons:
        print(f"\n🎯 Evaluando Horizonte Objetivo: [{target_col}]")
        
        y = pd.to_numeric(df[target_col], errors="coerce").to_numpy()
        valid_idx = ~pd.isna(y)
        X_target = X.loc[valid_idx]
        y_target = y[valid_idx]

        # Control para evitar errores si hay muy pocas filas
        if len(X_target) < 10:
            print(f"   ⚠️ Muy pocas filas para {target_col}, saltando...")
            continue

        rmse_scores = []
        for train_idx, val_idx in tscv.split(X_target):
            X_tr, X_val = X_target.iloc[train_idx], X_target.iloc[val_idx]
            y_tr, y_val = y_target[train_idx], y_target[val_idx]

            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr)

            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            rmse_scores.append(rmse)

        mean_rmse = np.mean(rmse_scores)
        comparison_results[target_col] = mean_rmse
        print(f"   -> RMSE Promedio para {target_col}: {mean_rmse:.5f}")

        final_model = XGBRegressor(**params)
        final_model.fit(X_target, y_target)
        
        model_path = os.path.join(MODEL_OUTPUT_DIR, f"xgboost_model_{target_col}.json")
        final_model.save_model(model_path)
        print(f"   💾 Modelo guardado en: {model_path}")

    print("\n" + "="*70)
    print("🏆 TABLA COMPARATIVA DE RENDIMIENTO (RMSE por Horizonte)")
    print("="*70)
    sorted_comparison = sorted(comparison_results.items(), key=lambda x: x[1])
    for horizon, rmse in sorted_comparison:
        print(f"  * {horizon:10} | RMSE Promedio: {rmse:.5f}")
    print("="*70)
    print("✅ ¡Entrenamiento completado!\n")

if __name__ == "__main__":
    train_and_compare_multi_horizon()
