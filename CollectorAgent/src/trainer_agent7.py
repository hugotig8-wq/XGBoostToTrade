import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

# Rutas absolutas del entorno
DATASET_PATH = "/home/humberto/master_ml_dataset.parquet"
MODEL_OUTPUT_DIR = "/home/humberto/"

def load_best_parameters(is_classification=False) -> dict:
    """Carga los hiperparámetros optimizados."""
    params = {
        "n_estimators": 200,        
        "max_depth": 4,
        "gamma": 0,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    if is_classification:
        params["eval_metric"] = "logloss"
        params["objective"] = "binary:logistic"
    else:
        params["eval_metric"] = "rmse"
        params["objective"] = "reg:squarederror"
    return params


def load_dataset(parquet_path: str = DATASET_PATH):
    """Carga el dataset maestro y separa las características."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"❌ No se encontró el dataset maestro en: {parquet_path}")
        
    df = pd.read_parquet(parquet_path)
    if "datetime" in df.columns:
        df = df.sort_values("datetime")

    ignore_cols = [
        "date", "datetime", "ticker", "published_at", "published_at_dt", "source_url",
        "forward_max_high", "forward_max_profit", "target_high_profit",
        "return_1h", "return_3h", "return_6h", "return_12h", "return_1d",
        "return_2d", "return_3d", "return_5d", "return_10d",
        "source", "event_actor", "subject_entity", "sector", 
        "event_type", "relationship_reason", "return_status", "price_timestamp",
        "price_before_news"
    ]
    
    numeric_cols = df.select_dtypes(include=[np.number, "float", "int", "bool"]).columns.tolist()
    features = [c for c in numeric_cols if c not in ignore_cols]

    X = df[features]
    return df, X, features


def train_and_compare_multi_horizon():
    print(f"🚀 Cargando dataset maestro desde: {DATASET_PATH}")
    df, X, features = load_dataset()
    print(f"📊 Dataset base: {len(X):,} filas | {len(features)} características válidas")

    # TimeSeriesSplit ajustado para datos pequeños
    tscv = TimeSeriesSplit(n_splits=2)
    
    reg_results = {}
    clf_results = {}

    print("\n" + "="*70)
    print("📈 INICIANDO ENTRENAMIENTO NATIVO CON NAN (Regresión y Clasificación)")
    print("="*70)

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    horizons = ["return_1h", "return_3h", "return_6h", "return_12h", "return_1d"]
    available_horizons = [h for h in horizons if h in df.columns]

    if not available_horizons:
        print("❌ No se encontraron columnas de retorno en el dataset.")
        return

    for target_col in available_horizons:
        print(f"\n🎯 Evaluando Horizonte Objetivo: [{target_col}]")
        
        # Limpiar nulos del target (XGBoost no puede predecir un target NaN)
        y_reg = pd.to_numeric(df[target_col], errors="coerce").to_numpy()
        valid_idx = ~pd.isna(y_reg)
        
        X_target = X.loc[valid_idx].copy()
        y_reg_target = y_reg[valid_idx]

        if len(X_target) < 6:
            print(f"   ⚠️ Muy pocas filas para {target_col} ({len(X_target)}). Saltando...")
            continue

        # ---------------------------------------------------------
        # 1. REGRESIÓN (XGBoost maneja los NaN de X nativamente)
        # ---------------------------------------------------------
        rmse_scores = []
        for train_idx, val_idx in tscv.split(X_target):
            X_tr, X_val = X_target.iloc[train_idx], X_target.iloc[val_idx]
            y_tr, y_val = y_reg_target[train_idx], y_reg_target[val_idx]

            model_reg = XGBRegressor(**load_best_parameters(is_classification=False))
            # Pasamos X_tr con sus NaN, XGBoost se encarga
            model_reg.fit(X_tr, y_tr)

            preds = model_reg.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            rmse_scores.append(rmse)

        mean_rmse = np.mean(rmse_scores)
        reg_results[target_col] = mean_rmse
        print(f"   📉 Regresión -> RMSE Promedio: {mean_rmse:.5f}")

        # Entrenar modelo final de regresión
        final_reg = XGBRegressor(**load_best_parameters(is_classification=False))
        final_reg.fit(X_target, y_reg_target)
        final_reg.save_model(os.path.join(MODEL_OUTPUT_DIR, f"reg_xgboost_{target_col}.json"))

        # Extraer las 5 características más importantes según XGBoost
        importances = final_reg.feature_importances_
        top_indices = np.argsort(importances)[-5:][::-1]
        top_feats_reg = [features[i] for i in top_indices]
        print(f"   🔍 Top 5 Características (Regresión): {top_feats_reg}")

        # ---------------------------------------------------------
        # 2. CLASIFICACIÓN (XGBoost maneja los NaN de X nativamente)
        # ---------------------------------------------------------
        y_clf_target = (y_reg_target > 0).astype(int)
        
        if len(np.unique(y_clf_target)) < 2:
            print(f"   ⚠️ Todos los retornos son iguales (solo hay una clase). Saltando clasificación...")
            continue

        acc_scores = []
        f1_scores = []
        
        for train_idx, val_idx in tscv.split(X_target):
            X_tr, X_val = X_target.iloc[train_idx], X_target.iloc[val_idx]
            y_tr, y_val = y_clf_target[train_idx], y_clf_target[val_idx]

            model_clf = XGBClassifier(**load_best_parameters(is_classification=True))
            model_clf.fit(X_tr, y_tr)

            preds = model_clf.predict(X_val)
            acc_scores.append(accuracy_score(y_val, preds))
            f1_scores.append(f1_score(y_val, preds, zero_division=0))

        mean_acc = np.mean(acc_scores)
        mean_f1 = np.mean(f1_scores)
        clf_results[target_col] = {"Accuracy": mean_acc, "F1": mean_f1}
        
        print(f"   📈 Clasificación -> Accuracy: {mean_acc:.3f} | F1-Score: {mean_f1:.3f}")

        # Entrenar modelo final de clasificación
        final_clf = XGBClassifier(**load_best_parameters(is_classification=True))
        final_clf.fit(X_target, y_clf_target)
        final_clf.save_model(os.path.join(MODEL_OUTPUT_DIR, f"clf_xgboost_{target_col}.json"))
        
        # Extraer las 5 características más importantes según XGBoost
        importances_clf = final_clf.feature_importances_
        top_indices_clf = np.argsort(importances_clf)[-5:][::-1]
        top_feats_clf = [features[i] for i in top_indices_clf]
        print(f"   🔍 Top 5 Características (Clasificación): {top_feats_clf}")


    # ---------------------------------------------------------
    # 3. TABLAS COMPARATIVAS FINALES
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print("🏆 TABLA COMPARATIVA: REGRESIÓN (RMSE por Horizonte)")
    print("="*70)
    sorted_reg = sorted(reg_results.items(), key=lambda x: x[1])
    for horizon, rmse in sorted_reg:
        print(f"  * {horizon:10} | RMSE Promedio: {rmse:.5f}")
    
    print("\n" + "="*70)
    print("🏆 TABLA COMPARATIVA: CLASIFICACIÓN (Sube/Baja por Horizonte)")
    print("="*70)
    sorted_clf = sorted(clf_results.items(), key=lambda x: x[1]["Accuracy"], reverse=True)
    for horizon, metrics in sorted_clf:
        print(f"  * {horizon:10} | Accuracy: {metrics['Accuracy']:.3f} | F1-Score: {metrics['F1']:.3f}")
    
    print("="*70)
    print("✅ ¡Entrenamiento Nativo con NaN completado con éxito!\n")


if __name__ == "__main__":
    train_and_compare_multi_horizon()
