"""
Agente Optimizador de Hiperparámetros con TimeSeriesSplit.
Evalúa XGBoost Classifier mediante probabilidades de la clase positiva [:, 1].
"""

import optuna
import polars as pl
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

# Silenciar mensajes informativos de Optuna para mantener la consola limpia
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Ruta absoluta al dataset
DATASET_PATH = "/workspaces/XGBoostToTrade/ColectorAgent/data/parquet/ml_ready_dataset_enhanced.parquet"


def load_dataset(parquet_path: str = DATASET_PATH):
    """Carga el dataset parquet ordenado por fecha y separa las características."""
    df = pl.read_parquet(parquet_path).sort("date")

    # Columnas que no deben ser usadas como predictoras
    ignore_cols = [
        "date", "ticker", "forward_max_high", 
        "forward_max_profit", "target_high_profit"
    ]
    features = [c for c in df.columns if c not in ignore_cols]

    X = df.select(features).to_pandas()
    y = df.select("target_high_profit").to_numpy().ravel()

    return features, X, y


def objective_cv(trial, X, y):
    """Función objetivo que evalúa los hiperparámetros con TimeSeriesSplit (5 pliegues)."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 350),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "gamma": trial.suggest_float("gamma", 0.5, 5.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.8, 2.0),
        "random_state": 42,
        "eval_metric": "logloss"
    }

    # Validación cruzada respetando la línea temporal
    tscv = TimeSeriesSplit(n_splits=5)
    auc_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)

        # Extraer únicamente la probabilidad de la clase positiva (1) -> [:, 1]
        preds_proba = model.predict_proba(X_val)[:, 1]
        
        # Calcular el AUC-ROC sobre el bloque de validación
        score = roc_auc_score(y_val, preds_proba)
        auc_scores.append(score)

    return np.mean(auc_scores)


def main():
    print(f"🚀 Cargando dataset desde: {DATASET_PATH}")
    features, X, y = load_dataset()
    print(f"📊 Dataset cargado: {len(X)} filas | {len(features)} características")

    print("\n🔎 AGENTE OPTIMIZANDO CON TIME-SERIES CROSS-VALIDATION (5 Folds)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_cv(trial, X, y), n_trials=40)

    print("\n" + "="*55)
    print(f"🏆 MEJOR AUC-ROC PROMEDIO EN VALIDACIÓN CRUZADA: {study.best_value:.4f}")
    print("\nMejores Hiperparámetros Encontrados:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("="*55)


if __name__ == "__main__":
    main()
    
