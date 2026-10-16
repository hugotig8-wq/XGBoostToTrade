"""
Agente de Predicción con Diagnóstico de Features (predict_agent.py)
------------------------------------------------------------------
Audita las características de entrada para asegurar que cada activo 
reciba un vector único y el modelo no colapse en predicciones constantes.
"""

import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

DATASET_PATH = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/master_ml_dataset.parquet"
MODELS_DIR = "/workspaces/XGBoostToTrade/CollectorAgent/data/models/"
OUTPUT_JSON_PATH = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/live_predictions.json"


def load_trained_models():
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
    print("🔮 Generando pronósticos con auditoría de características...")
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ No se encontró el dataset maestro en: {DATASET_PATH}")
        return
        
    df = pd.read_parquet(DATASET_PATH)
    if "date" in df.columns:
        df = df.sort_values("datetime" if "datetime" in df.columns else "date")
        
    models = load_trained_models()
    if not models:
        print("❌ No hay modelos entrenados disponibles.")
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
    print("🔍 AUDITORÍA DE ENTRADA Y PREDICCIONES MULTI-HORIZONTE")
    print("="*80)

    for idx, row in latest_candles.reset_index(drop=True).iterrows():
        ticker = row["ticker"]
        current_close = float(row["close"])
        wave_phase = str(row.get("elliott_wave_phase", "N/A"))
        rsi_val = float(row.get("rsi_14", 0.0))
        rel_vol = float(row.get("relative_volume", 0.0))
        
        asset_preds = {
            "ticker": ticker,
            "current_close": current_close,
            "elliott_wave_phase": wave_phase,
            "rsi_14": rsi_val,
            "relative_volume": rel_vol,
            "horizons": {}
        }
        
        print(f"\n📊 Activo: [{ticker}] | Precio: {current_close:.2f} | RSI(14): {rsi_val:.2f} | Vol. Rel: {rel_vol:.2f}")
        print(f"   🌊 Fase Estructural: {wave_phase}")
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

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions_payload, f, indent=4, ensure_ascii=False)

    print("\n" + "="*80)
    print("✅ Auditoría completada. Revisa si el RSI y el Volumen Relativo cambian por activo.")
    print("="*80 + "\n")


if __name__ == "__main__":
    my_watchlist = ["IBE.MC", "NEE", "GEV", "CVX", "XOM", "WTI", "BZ=F"]
    generate_live_predictions(watchlist_tickers=my_watchlist)
    
