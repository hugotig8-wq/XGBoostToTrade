"""
Módulo de Integración Multidimensional de Noticias y Mercado (news_market_integrator.py)
----------------------------------------------------------------------------------------
Fusiona los datos raw de precios intradiarios con las noticias estructuradas,
calcula indicadores técnicos avanzados, proyecta niveles de Fibonacci/Elliott 
y determina stop loss y take profit en función de la estructura de mercado.
"""

import os
import numpy as np
import pandas as pd


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la batería completa de indicadores técnicos sobre las velas horarias."""
    df = df.sort_values(by=["ticker", "datetime"]).copy()
    
    # Agrupamos por ticker para calcular los indicadores de forma independiente
    groups = []
    for ticker, group in df.groupby("ticker"):
        g = group.copy()
        
        # Medias Móviles
        g["sma_20"] = g["close"].rolling(window=20).mean()
        g["ema_20"] = g["close"].ewm(span=20, adjust=False).mean()
        g["ema_50"] = g["close"].ewm(span=50, adjust=False).mean()
        g["ema_200"] = g["close"].ewm(span=200, adjust=False).mean()
        
        # MACD
        exp12 = g["close"].ewm(span=12, adjust=False).mean()
        exp26 = g["close"].ewm(span=26, adjust=False).mean()
        g["macd"] = exp12 - exp26
        g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
        g["macd_histogram"] = g["macd"] - g["macd_signal"]
        
        # RSI 14
        delta = g["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        g["rsi_14"] = 100 - (100 / (1 + rs))
        
        # Bandas de Bollinger (20, 2)
        bb_std = g["close"].rolling(window=20).std()
        g["bb_upper"] = g["sma_20"] + (bb_std * 2)
        g["bb_lower"] = g["sma_20"] - (bb_std * 2)
        g["bb_width"] = (g["bb_upper"] - g["bb_lower"]) / (g["sma_20"] + 1e-9)
        g["bb_percent_b"] = (g["close"] - g["bb_lower"]) / (g["bb_upper"] - g["bb_lower"] + 1e-9)
        
        # ATR 14 y ATR %
        high_low = g["high"] - g["low"]
        high_close = (g["high"] - g["close"].shift()).abs()
        low_close = (g["low"] - g["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        g["atr_14"] = true_range.rolling(window=14).mean()
        g["atr_pct"] = g["atr_14"] / (g["close"] + 1e-9)
        
        # OBV (On-Balance Volume)
        g["obv"] = (np.sign(g["close"].diff()) * g["volume"]).fillna(0).cumsum()
        
        # VWAP y Price to VWAP
        q = g["volume"]
        typical_price = (g["high"] + g["low"] + g["close"]) / 3
        g["vwap"] = (typical_price * q).cumsum() / (q.cumsum() + 1e-9)
        g["price_to_vwap"] = g["close"] / (g["vwap"] + 1e-9)
        
        # ADX 14, +DI, -DI
        plus_dm = g["high"].diff()
        minus_dm = g["low"].diff()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
        
        tr_smooth = true_range.rolling(window=14).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=14).mean() / (tr_smooth + 1e-9)
        minus_di = 100 * pd.Series(minus_dm).rolling(window=14).mean() / (tr_smooth + 1e-9)
        g["plus_di"] = plus_di
        g["minus_di"] = minus_di
        dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)
        g["adx_14"] = dx.rolling(window=14).mean()
        
        # Estocástico (%K, %D)
        lowest_low = g["low"].rolling(window=14).min()
        highest_high = g["high"].rolling(window=14).max()
        g["%k"] = 100 * ((g["close"] - lowest_low) / (highest_high - lowest_low + 1e-9))
        g["%d"] = g["%k"].rolling(window=3).mean()
        
        # Volumen Relativo (respecto a la media de 20 periodos)
        g["relative_volume"] = g["volume"] / (g["volume"].rolling(window=20).mean() + 1e-9)
        
        groups.append(g)
        
    return pd.concat(groups, ignore_index=True)


def analyze_market_structure_and_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta máximos y mínimos relativos para identificar ondas de Elliott,
    niveles de Fibonacci, y calcular Stop Loss / Take Profit.
    """
    groups = []
    for ticker, group in df.groupby("ticker"):
        g = group.copy()
        
        # Identificar pivotes locales (ventana de 5 periodos)
        window = 5
        g["is_high"] = g["high"] == g["high"].rolling(window=window*2+1, center=True).max()
        g["is_low"] = g["low"] == g["low"].rolling(window=window*2+1, center=True).min()
        
        # Calcular Retrocesos de Fibonacci y Ondas basadas en los últimos extremos detectados
        # Estimación simplificada de ondas y niveles para el motor cuantitativo
        rolling_max = g["high"].rolling(window=50).max()
        rolling_min = g["low"].rolling(window=50).min()
        diff = rolling_max - rolling_min
        
        # Niveles de Fibonacci clave
        g["fib_236"] = rolling_max - (diff * 0.236)
        g["fib_382"] = rolling_max - (diff * 0.382)
        g["fib_500"] = rolling_max - (diff * 0.500)
        g["fib_618"] = rolling_max - (diff * 0.618)
        
        # Asignación heurística de Ondas de Elliott según posición en el rango
        g["elliott_wave_phase"] = np.where(g["close"] > g["fib_382"], "Impulse_Wave_3_or_5", "Correction_Wave_2_or_4")
        
        # Cálculo dinámico de Stop Loss y Take Profit basados en ATR y Soportes/Resistencias
        atr = g["atr_14"].fillna(g["close"] * 0.01)
        g["stop_loss"] = np.where(g["elliott_wave_phase"].str.contains("Impulse"), g["close"] - (2 * atr), g["close"] - (1.5 * atr))
        g["take_profit"] = np.where(g["elliott_wave_phase"].str.contains("Impulse"), g["close"] + (3.5 * atr), g["close"] + (2.5 * atr))
        
        groups.append(g)
        
    return pd.concat(groups, ignore_index=True)


def integrate_news_and_market(
    raw_parquet: str = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/raw/financial_data.parquet",
    news_csv: str = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/cleaned_sentiment_features.parquet",
    output_parquet: str = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/master_ml_dataset.parquet"
) -> None:
    """Ejecuta el pipeline completo de integración entre velas raw, noticias y métricas cuantitativas."""
    print("🔄 Iniciando integración avanzada de mercado y noticias...")
    
    if not os.path.exists(raw_parquet):
        print(f"❌ No se encontró el archivo de precios raw en: {raw_parquet}")
        return
        
    df_market = pd.read_parquet(raw_parquet)
    df_market["datetime"] = pd.to_datetime(df_market["datetime"])
    df_market["date"] = df_market["datetime"].dt.date
    
    # 1. Calcular indicadores técnicos
    print("📈 Calculando indicadores técnicos sobre velas de 1h...")
    df_tech = compute_technical_indicators(df_market)
    
    # 2. Calcular estructura de mercado, Fibonacci, Elliott, SL y TP
    print("📐 Mapeando ondas de Elliott, Fibonacci, Stop Loss y Take Profit...")
    df_structured = analyze_market_structure_and_levels(df_tech)
    
    # 3. Cruzar con noticias si el archivo existe
    if os.path.exists(news_csv):
        print(f"📰 Cruzando datos con el repositorio de noticias: {news_csv}")
        # Soporta tanto CSV como Parquet para las noticias
        df_news = pd.read_parquet(news_csv) if news_csv.endswith(".parquet") else pd.read_csv(news_csv)
        if "published_at" in df_news.columns:
            df_news["date"] = pd.to_datetime(df_news["published_at"]).dt.date
            
            # Merge inteligente por fecha y activo afectado
            df_final = pd.merge(
                df_structured, 
                df_news, 
                how="left", 
                left_on=["date", "ticker"], 
                right_on=["date", "affected_assets"], 
                suffixes=("_market", "_news")
            )
        else:
            df_final = df_structured
    else:
        print("⚠️ No se encontró archivo de noticias externo. Consolidando solo matriz técnica y estructural.")
        df_final = df_structured
        
    # Guardar dataset maestro optimizado para XGBoost
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df_final.to_parquet(output_parquet, engine="pyarrow", compression="snappy", index=False)
    
    print("\n" + "="*60)
    print("✅ INTEGRACIÓN MAESTRA COMPLETADA CON ÉXITO")
    print("="*60)
    print(f" Total de registros procesados : {len(df_final):,}")
    print(f" Dataset guardado en           : {output_parquet}")
    print("="*60 + "\n")


if __name__ == "__main__":
    integrate_news_and_market()
      
