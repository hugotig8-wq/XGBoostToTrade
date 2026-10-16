"""
Agente de Predicción y Señales Operativas (predict_agent.py)
------------------------------------------------------------
Carga los modelos XGBoost entrenados para cada horizonte temporal, 
procesa los datos recientes de la watchlist y genera pronósticos de retornos,
fases de Elliott y niveles de Stop Loss / Take Profit.
"""

import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Rutas del entorno
DATASET_PATH = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/master_ml_dataset.parquet"
MODELS_DIR = "/workspaces/XGBoostToTrade/CollectorAgent/data/models/"


def load_trained_models() -> dict:
    """Carga todos los modelos XGBoost multi-horizonte guardados."""
    horizons = ["return_1h", "return_3h", "return_6h", "return_12h", "return_1d"]
    models = {}
    
    for h in horizons:
        model_path = os.path.join(MODELS_DIR, f"xgboost_model_{h}.json")
        if os.path.exists(model_path):
            model = XGBRegressor()
            model.load_model(model_path)
            models[h] = model
        else:
            print(f"⚠️ No se encontró el modelo para {h} en {model_path}")
            
    return models


def generate_live_predictions(watchlist_tickers=None):
    """Genera pronósticos operativos para los activos de la watchlist."""
    print("🔮 Iniciando generación de pronósticos operativos...")
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ No se encontró el dataset maestro en: {DATASET_PATH}")
        return
        
    df = pd.read_parquet(DATASET_PATH)
    if "date" in df.columns:
        df = df.sort_values("datetime" if "datetime" in df.columns else "date")
        
    models = load_trained_models()
    if not models:
        print("❌ No hay modelos entrenados disponibles para predecir.")
        return
        
    # Filtrar por tickers de interés si se especifican
    if watchlist_tickers:
        df = df[df["ticker"].isin(watchlist_tickers)]
        
    # Obtener exclusivamente la última vela (más reciente) de cada activo
    latest_candles = df.groupby("ticker").tail(1).copy()
    
    # Definir las características utilizadas en el entrenamiento (excluyendo metadatos y retornos)
    ignore_cols = [
        "date", "datetime", "ticker", "published_at", "source_url",
        "forward_max_high", "forward_max_profit", "target_high_profit",
        "return_1h", "return_3h", "return_6h", "return_12h", "return_1d",
        "elliott_wave_phase"
    ]
    numeric_cols = latest_candles.select_dtypes(include=[np.number, "float", "int", "bool"]).columns.tolist()
    features = [c for c in numeric_cols if c not in ignore_cols]
    
    X_live = latest_candles[features]

    print("\n" + "="*80)
    print("🎯 SEÑALES OPERATIVAS Y PRONÓSTICOS MULTI-HORIZONTE (TIEMPO REAL)")
    print("="*80)

    for idx, row in latest_candles.reset_index(drop=True).iterrows():
        ticker = row["ticker"]
        current_close = row["close"]
        wave_phase = row.get("elliott_wave_phase", "N/A")
        stop_loss = row.get("stop_loss", current_close * 0.98)
        take_profit = row.get("take_profit", current_close * 1.02)
        
        print(f"\n📊 Activo: [{ticker}] | Precio Actual: {current_close:.2f}")
        print(f"   🌊 Fase Estructural (Elliott): {wave_phase}")
        print(f"   🛑 Stop Loss Sugerido  : {stop_loss:.2f}")
        print(f"   🎯 Take Profit Sugerido: {take_profit:.2f}")
        print("   📈 Retornos Proyectados:")
        
        # Extraer características de esta fila específica
        x_sample = X_live.iloc[[idx]]
        
        for horizon, model in models.items():
            pred_return = model.predict(x_sample)[0]
            direction = "🟢 ALCISTA" if pred_return > 0 else "🔴 BAJISTA"
            print(f"      * {horizon:10} -> {pred_return*100:+6.2f}% ({direction})")
            
    print("\n" + "="*80)
    print("✅ PRONÓSTICOS GENERADOS CON ÉXITO")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Watchlist definida en tu arquitectura
    my_watchlist = ["IBE.MC", "NEE", "GEV", "CVX", "XOM", "WTI", "BZ=F"]
    generate_live_predictions(watchlist_tickers=my_watchlist)
          
