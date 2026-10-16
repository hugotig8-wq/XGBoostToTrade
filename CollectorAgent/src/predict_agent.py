"""
Agente de Predicción y Señales Operativas con Exportación a JSON (predict_agent.py)
----------------------------------------------------------------------------------
Carga los modelos XGBoost, genera pronósticos y los guarda en un archivo JSON 
ligero para consulta instantánea sin consumo excesivo de CPU en Codespaces.
"""

import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Rutas del entorno
DATASET_PATH = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/master_ml_dataset.parquet"
MODELS_DIR = "/workspaces/XGBoostToTrade/CollectorAgent/data/models/"
OUTPUT_JSON_PATH = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/live_predictions.json"


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
            
    return models


def generate_live_predictions(watchlist_tickers=None):
    """Genera pronósticos operativos y los exporta a un archivo JSON ultraligero."""
    print("🔮 Generando pronósticos operativos y guardando en JSON...")
    
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
        
    if watchlist_tickers:
        df = df[df["ticker"].isin(watchlist_tickers)]
        
    latest_candles = df.groupby("ticker").tail(1).copy()
    
    ignore_cols = [
        "date", "datetime", "ticker", "published_at", "source_url",
        "forward_max_high", "forward_max_profit", "target_high_profit",
        "return_1h", "return_3h", "return_6h", "return_12h", "return_1d",
        "elliott_wave_phase"
    ]
    numeric_cols = latest_candles.select_dtypes(include=[np.number, "float", "int", "bool"]).columns.tolist()
    features = [c for c in numeric_cols if c not in ignore_cols]
    
    X_live = latest_candles[features]
    predictions_payload = []

    print("\n" + "="*80)
    print("🎯 SEÑALES OPERATIVAS MULTI-HORIZONTE (EXPORTADAS A JSON)")
    print("="*80)

    for idx, row in latest_candles.reset_index(drop=True).iterrows():
        ticker = row["ticker"]
        current_close = float(row["close"])
        wave_phase = str(row.get("elliott_wave_phase", "N/A"))
        stop_loss = float(row.get("stop_loss", current_close * 0.98))
        take_profit = float(row.get("take_profit", current_close * 1.02))
        
        asset_preds = {
            "ticker": ticker,
            "current_close": current_close,
            "elliott_wave_phase": wave_phase,
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "horizons": {}
        }
        
        print(f"\n📊 Activo: [{ticker}] | Precio Actual: {current_close:.2f}")
        print(f"   🌊 Fase Estructural: {wave_phase}")
        print(f"   🛑 Stop Loss: {stop_loss:.2f} | 🎯 Take Profit: {take_profit:.2f}")
        print("   📈 Retornos Proyectados:")
        
        x_sample = X_live.iloc[[idx]]
        
        for horizon, model in models.items():
            pred_return = float(model.predict(x_sample)[0])
            direction = "ALCISTA" if pred_return > 0 else "BAJISTA"
            asset_preds["horizons"][horizon] = {
                "predicted_return_pct": round(pred_return * 100, 2),
                "direction": direction
            }
            print(f"      * {horizon:10} -> {pred_return*100:+6.2f}% ({direction})")
            
        predictions_payload.append(asset_preds)

    # Guardar en JSON para lectura instantánea sin consumo de cómputo
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions_payload, f, indent=4, ensure_ascii=False)

    print("\n" + "="*80)
    print(f"💾 Predicciones guardadas exitosamente en: {OUTPUT_JSON_PATH}")
    print("="*80 + "\n")


if __name__ == "__main__":
    my_watchlist = ["IBE.MC", "NEE", "GEV", "CVX", "XOM", "WTI", "BZ=F"]
    generate_live_predictions(watchlist_tickers=my_watchlist)
            
