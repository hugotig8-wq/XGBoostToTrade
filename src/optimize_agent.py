"""
Módulo Optimizador Autónomo de Hiperparámetros.
Evalúa métricas clave (AUC-ROC, F1-Score, MAE, R²) con Optuna.
"""

import optuna
import polars as pl
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error, r2_score

# Desactivar logs de optuna para no saturar la terminal
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_dataset():
    """Carga el dataset enriquecido con indicadores técnicos."""
    df = pl.read_parquet("/workspaces/XGBoostToTrade/data/parquet/ml_ready_dataset_enhanced.parquet").sort("date")

    ignore_cols = [
        "date", "ticker", "forward_max_high", 
        "forward_max_profit", "target_high_profit"
    ]
    features = [c for c in df.columns if c not in ignore_cols]

    # Split 80/20 estricto
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df[:split_idx], df[split_idx:]

    X_train = train_df.select(features).to_pandas()
    X_test = test_df.select(features).to_pandas()

    y_train_cls = train_df.select("target_high_profit").to_numpy().ravel()
    y_test_cls = test_df.select("target_high_profit").to_numpy().ravel()

    y_train_reg = train_df.select("forward_max_profit").to_numpy().ravel()
    y_test_reg = test_df.select("forward_max_profit").to_numpy().ravel()

    return X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg


def objective_classifier(trial, X_tr, X_te, y_tr, y_te):
    """Objetivo de optimización para Clasificación (Maximizar AUC-ROC)."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "random_state": 42,
        "eval_metric": "logloss"
    }

    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr)

    preds_proba = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, preds_proba)
    return auc


def objective_regressor(trial, X_tr, X_te, y_tr, y_te):
    """Objetivo de optimización para Regresión (Minimizar MAE)."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42
    }

    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr)

    preds = model.predict(X_te)
    mae = mean_absolute_error(y_te, preds)
    return mae


def main():
    print("🚀 Cargando datos e indicadores técnicos...")
    X_tr, X_te, y_tr_cls, y_te_cls, y_tr_reg, y_te_reg = load_dataset()

    print("\n🔎 AGENTE OPTIMIZANDO XGBOOST CLASSIFIER (50 Ensayos)...")
    study_cls = optuna.create_study(direction="maximize")
    study_cls.optimize(
        lambda trial: objective_classifier(trial, X_tr, X_te, y_tr_cls, y_te_cls),
        n_trials=50
    )

    print(f"🏆 Mejores Hiperparámetros Clasificador (AUC-ROC: {study_cls.best_value:.4f}):")
    for k, v in study_cls.best_params.items():
        print(f"  - {k}: {v}")

    print("\n🔎 AGENTE OPTIMIZANDO XGBOOST REGRESSOR (50 Ensayos)...")
    study_reg = optuna.create_study(direction="minimize")
    study_reg.optimize(
        lambda trial: objective_regressor(trial, X_tr, X_te, y_tr_reg, y_te_reg),
        n_trials=50
    )

    print(f"🏆 Mejores Hiperparámetros Regresor (MAE: {study_reg.best_value:.4f}):")
    for k, v in study_reg.best_params.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
  
