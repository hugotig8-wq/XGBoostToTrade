import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.impute import SimpleImputer

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

    # Columnas estrictamente prohibidas para evitar Data Leakage y errores de XGBoost
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

    # A. TimeSeriesSplit ajustado para datos pequeños
    tscv = TimeSeriesSplit(n_splits=2)
    
    reg_results = {}
    clf_results = {}

    print("\n" + "="*70)
    print("📈 INICIANDO ENTRENAMIENTO MULTI-HORIZONTE (Regresión y Clasificación)")
    print("="*70)

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    horizons = ["return_1h", "return_3h", "return_6h", "return_12h", "return_1d"]
    available_horizons = [h for h in horizons if h in df.columns]

    if not available_horizons:
        print("❌ No se encontraron columnas de retorno en el dataset.")
        return

    for target_col in available_horizons:
        print(f"\n🎯 Evaluando Horizonte Objetivo: [{target_col}]")
        
        # Limpiar nulos del target
        y_reg = pd.to_numeric(df[target_col], errors="coerce").to_numpy()
        valid_idx = ~pd.isna(y_reg)
        
        X_target = X.loc[valid_idx].copy()
        y_reg_target = y_reg[valid_idx]

        # Comprobar que hay suficientes datos para el split
        if len(X_target) < 6:
            print(f"   ⚠️ Muy pocas filas para {target_col} ({len(X_target)}). Saltando...")
            continue

        # ---------------------------------------------------------
        # 1. REGRESIÓN (Predecir el retorno exacto)
        # ---------------------------------------------------------
        rmse_scores = []
        for train_idx, val_idx in tscv.split(X_target):
            X_tr, X_val = X_target.iloc[train_idx], X_target.iloc[val_idx]
            y_tr, y_val = y_reg_target[train_idx], y_reg_target[val_idx]

            # Imputar NaNs (SelectKBest no los acepta)
            imputer_reg = SimpleImputer(strategy='median')
            X_tr_imp = imputer_reg.fit_transform(X_tr)
            X_val_imp = imputer_reg.transform(X_val)

            # B. Feature Selection para regresión
            k_best = min(5, X_tr_imp.shape[1])
            selector = SelectKBest(score_func=f_regression, k=k_best)
            X_tr_sel = selector.fit_transform(X_tr_imp, y_tr)
            X_val_sel = selector.transform(X_val_imp)

            model_reg = XGBRegressor(**load_best_parameters(is_classification=False))
            model_reg.fit(X_tr_sel, y_tr)

            preds = model_reg.predict(X_val_sel)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            rmse_scores.append(rmse)

        mean_rmse = np.mean(rmse_scores)
        reg_results[target_col] = mean_rmse
        print(f"   📉 Regresión -> RMSE Promedio: {mean_rmse:.5f}")

        # Entrenar modelo final de regresión
        imputer_final_reg = SimpleImputer(strategy='median')
        X_final_reg_imp = imputer_final_reg.fit_transform(X_target)
        
        selector_reg = SelectKBest(score_func=f_regression, k=min(5, X_target.shape[1]))
        X_final_reg_sel = selector_reg.fit_transform(X_final_reg_imp, y_reg_target)
        
        final_reg = XGBRegressor(**load_best_parameters(is_classification=False))
        final_reg.fit(X_final_reg_sel, y_reg_target)
        final_reg.save_model(os.path.join(MODEL_OUTPUT_DIR, f"reg_xgboost_{target_col}.json"))

        # ---------------------------------------------------------
        # 2. CLASIFICACIÓN (Predecir si sube o baja)
        # ---------------------------------------------------------
        # C. Crear target binario: 1 si el retorno es positivo, 0 si es negativo
        y_clf_target = (y_reg_target > 0).astype(int)
        
        if len(np.unique(y_clf_target)) < 2:
            print(f"   ⚠️ Todos los retornos son iguales (solo hay una clase). Saltando clasificación...")
            continue

        acc_scores = []
        f1_scores = []
        
        for train_idx, val_idx in tscv.split(X_target):
            X_tr, X_val = X_target.iloc[train_idx], X_target.iloc[val_idx]
            y_tr, y_val = y_clf_target[train_idx], y_clf_target[val_idx]

            # Imputar NaNs
            imputer_clf = SimpleImputer(strategy='median')
            X_tr_imp = imputer_clf.fit_transform(X_tr)
            X_val_imp = imputer_clf.transform(X_val)

            # B. Feature Selection para clasificación
            k_best = min(5, X_tr_imp.shape[1])
            selector = SelectKBest(score_func=f_classif, k=k_best)
            X_tr_sel = selector.fit_transform(X_tr_imp, y_tr)
            X_val_sel = selector.transform(X_val_imp)

            model_clf = XGBClassifier(**load_best_parameters(is_classification=True))
            model_clf.fit(X_tr_sel, y_tr)

            preds = model_clf.predict(X_val_sel)
            acc_scores.append(accuracy_score(y_val, preds))
            f1_scores.append(f1_score(y_val, preds, zero_division=0))

        mean_acc = np.mean(acc_scores)
        mean_f1 = np.mean(f1_scores)
        clf_results[target_col] = {"Accuracy": mean_acc, "F1": mean_f1}
        
        print(f"   📈 Clasificación -> Accuracy: {mean_acc:.3f} | F1-Score: {mean_f1:.3f}")

        # Entrenar modelo final de clasificación
        imputer_final_clf = SimpleImputer(strategy='median')
        X_final_clf_imp = imputer_final_clf.fit_transform(X_target)
        
        selector_clf = SelectKBest(score_func=f_classif, k=min(5, X_target.shape[1]))
        X_final_clf_sel = selector_clf.fit_transform(X_final_clf_imp, y_clf_target)
        
        final_clf = XGBClassifier(**load_best_parameters(is_classification=True))
        final_clf.fit(X_final_clf_sel, y_clf_target)
        final_clf.save_model(os.path.join(MODEL_OUTPUT_DIR, f"clf_xgboost_{target_col}.json"))
        
        # Mostrar qué características salvó el SelectKBest
        selected_feats = X_target.columns[selector_clf.get_support()].tolist()
        print(f"   🔍 Top 5 características para {target_col}: {selected_feats}")


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
    print("✅ ¡Entrenamiento (Regresión y Clasificación) completado con éxito!\n")


if __name__ == "__main__":
    train_and_compare_multi_horizon()
