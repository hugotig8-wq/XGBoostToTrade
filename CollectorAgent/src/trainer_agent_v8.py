import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

# ==========================================
# 1. CONFIGURACIÓN INICIAL
# ==========================================
# Ruta de tu dataset (ajústala si es necesario)
DATASET_PATH = '/home/humberto/master_ml_dataset.parquet' 

# Horizontes a evaluar
horizontes = ['return_1h', 'return_3h', 'return_6h', 'return_12h', 'return_1d']

# Columnas que NO son features (metadatos y targets)
cols_excluir = ['datetime', 'ticker', 'asset', 'return_1h', 'return_3h', 'return_6h', 'return_12h', 'return_1d', 
                'return_2d', 'return_3d', 'return_5d', 'return_10d']

print("🚀 Cargando dataset...")
df = pd.read_parquet(DATASET_PATH)
features = [c for c in df.columns if c not in cols_excluir]

# ==========================================
# FUNCIÓN PARA ENCONTRAR EL MEJOR THRESHOLD
# ==========================================
def encontrar_mejor_threshold(y_true, y_probs):
    mejor_f1 = 0
    mejor_thresh = 0.5
    # Probamos umbrales del 0.1 al 0.9
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_probs >= thresh).astype(int)
        # Solo calculamos F1 si hay predicciones positivas para evitar warnings
        if len(np.unique(y_pred)) > 1:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > mejor_f1:
                mejor_f1 = f1
                mejor_thresh = thresh
    return mejor_thresh, mejor_f1

# ==========================================
# 2. BUCLE DE ENTRENAMIENTO
# ==========================================
for target in horizontes:
    print(f"\n==================================================")
    print(f"🎯 Evaluando Horizonte Objetivo: [{target}]")
    print(f"==================================================")
    
    # Limpieza de NaNs para este target específico
    df_target = df.dropna(subset=[target]).copy()
    
    # Convertir a Clasificación: 1 si el retorno es positivo, 0 si es negativo
    y_reg = df_target[target]
    y_clf = (y_reg > 0).astype(int)
    X = df_target[features]
    
    # División Train/Test (80/20)
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42, shuffle=False
    )

    # --------------------------------------------------
    # A) REGRESIÓN (Predecir el retorno exacto)
    # --------------------------------------------------
    print("📈 Entrenando Regresión...")
    params_reg = {
        'objective': 'reg:squarederror',
        'n_estimators': 500,       # 500 árboles para todos
        'learning_rate': 0.05,     # Tasa base
        'max_depth': 4,
        'random_state': 42
    }
    
    # El IF MÁGICO para 12h
    if target == 'return_12h':
        print("   ⚙️ Aplicando hiperparámetros específicos para 12h (Regresión)...")
        params_reg['learning_rate'] = 0.01 # Aprendizaje más lento
        params_reg['max_depth'] = 5        # Árboles más profundos
        params_reg['subsample'] = 0.8      # Prevenir sobreajuste
        params_reg['colsample_bytree'] = 0.8

    model_reg = xgb.XGBRegressor(**params_reg)
    model_reg.fit(X_train, y_reg_train)
    
    preds_reg = model_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_reg_test, preds_reg))
    print(f"   -> RMSE Promedio: {rmse:.5f}")

    # Guardar JSON de Regresión
    model_reg.save_model(f'reg_xgboost_{target}_v8.json')

    # --------------------------------------------------
    # B) CLASIFICACIÓN (Predecir si Sube o Baja)
    # --------------------------------------------------
    print("📊 Entrenando Clasificación...")
    params_clf = {
        'objective': 'binary:logistic',
        'n_estimators': 500,       # 500 árboles para todos
        'learning_rate': 0.05,
        'max_depth': 4,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    # El IF MÁGICO para 12h
    if target == 'return_12h':
        print("   ⚙️ Aplicando hiperparámetros específicos para 12h (Clasificación)...")
        params_clf['learning_rate'] = 0.01
        params_clf['max_depth'] = 5
        # Damos más peso a la clase positiva (subidas) porque estaba desbalanceado
        # (El base_score anterior era 0.34, así que hay más bajadas que subidas)
        params_clf['scale_pos_weight'] = 2.0 

    model_clf = xgb.XGBClassifier(**params_clf)
    model_clf.fit(X_train, y_clf_train)
    
    # Obtenemos probabilidades, no solo el 0 o 1
    probs_clf = model_clf.predict_proba(X_test)[:, 1]
    
    # Buscamos el mejor umbral para maximizar el F1
    best_thresh, best_f1 = encontrar_mejor_threshold(y_clf_test, probs_clf)
    
    # Aplicamos el mejor umbral
    preds_clf = (probs_clf >= best_thresh).astype(int)
    acc = accuracy_score(y_clf_test, preds_clf)
    
    print(f"   -> Mejor Threshold encontrado: {best_thresh:.2f}")
    print(f"   -> Accuracy: {acc:.3f} | F1-Score: {best_f1:.3f}")

    # Guardar JSON de Clasificación
    model_clf.save_model(f'clf_xgboost_{target}_v8.json')

print("\n✅ ¡Entrenamiento completado con éxito! Archivos guardados.")
