import os
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

def convert_csv_to_parquet(csv_path: str, parquet_path: str):
    """Convierte el CSV raw a Parquet para un procesamiento mucho más rápido."""
    if not os.path.exists(parquet_path):
        print(f"⚙️ Convirtiendo {csv_path} a Parquet...")
        df = pd.read_csv(csv_path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
    return parquet_path

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores técnicos asegurando que no haya data leakage."""
    df = df.sort_values(by=["ticker", "datetime"]).copy()
    df["date"] = df["datetime"].dt.date 
    
    groups = []
    for ticker, group in df.groupby("ticker"):
        g = group.copy()
        
        # 1. Medias Móviles
        g["sma_20"] = g["close"].rolling(window=20).mean()
        g["ema_20"] = g["close"].ewm(span=20, adjust=False).mean()
        g["ema_50"] = g["close"].ewm(span=50, adjust=False).mean()
        
        # 2. MACD
        exp12 = g["close"].ewm(span=12, adjust=False).mean()
        exp26 = g["close"].ewm(span=26, adjust=False).mean()
        g["macd"] = exp12 - exp26
        g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
        
        # 3. RSI 14
        delta = g["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        g["rsi_14"] = 100 - (100 / (1 + rs))
        
        # 4. Bandas de Bollinger
        bb_std = g["close"].rolling(window=20).std()
        g["bb_upper"] = g["sma_20"] + (bb_std * 2)
        g["bb_lower"] = g["sma_20"] - (bb_std * 2)
        g["bb_width"] = (g["bb_upper"] - g["bb_lower"]) / (g["sma_20"] + 1e-9)
        g["bb_percent_b"] = (g["close"] - g["bb_lower"]) / (g["bb_upper"] - g["bb_lower"] + 1e-9)
        
        # 5. ATR 14
        high_low = g["high"] - g["low"]
        high_close = (g["high"] - g["close"].shift()).abs()
        low_close = (g["low"] - g["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        g["atr_14"] = true_range.rolling(window=14).mean()
        g["atr_pct"] = g["atr_14"] / (g["close"] + 1e-9)
        
        # 6. OBV y Volumen Relativo
        g["obv"] = (np.sign(g["close"].diff()) * g["volume"]).fillna(0).cumsum()
        g["relative_volume"] = g["volume"] / (g["volume"].rolling(window=20).mean() + 1e-9)
        
        # 7. VWAP DIARIO (Corrección del error de Gemini)
        typical_price = (g["high"] + g["low"] + g["close"]) / 3
        vwap_daily = (typical_price * g["volume"]).groupby(g["date"]).cumsum() / (g["volume"].groupby(g["date"]).cumsum() + 1e-9)
        g["vwap"] = vwap_daily
        g["price_to_vwap"] = g["close"] / (g["vwap"] + 1e-9)
        
        groups.append(g)
    return pd.concat(groups, ignore_index=True)

def apply_elliott_and_levels(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Calcula Ondas de Elliott usando pivotes fractales y SL/TP estructurales."""
    col_suffix = f"_{window}"
    groups = []
    
    for ticker, group in df.groupby("ticker"):
        g = group.copy()
        
        # Encontrar máximos y mínimos locales (Pivotes) evitando Data Leakage
        high_values = g["high"].shift(1).ffill().values
        max_indices = argrelextrema(high_values, np.greater, order=window//2)[0]
        max_mask = np.zeros(len(g), dtype=bool)
        max_mask[max_indices] = True
        
        low_values = g["low"].shift(1).ffill().values
        min_indices = argrelextrema(low_values, np.less, order=window//2)[0]
        min_mask = np.zeros(len(g), dtype=bool)
        min_mask[min_indices] = True
        
        # Guardar el último pivote visto (swing high/low)
        g[f"last_swing_high{col_suffix}"] = g["high"].where(max_mask).ffill()
        g[f"last_swing_low{col_suffix}"] = g["low"].where(min_mask).ffill()
        
        # Rango del pivote
        g[f"swing_range{col_suffix}"] = g[f"last_swing_high{col_suffix}"] - g[f"last_swing_low{col_suffix}"]
        
        # Fuerza estructural (para la dominancia)
        g[f"wave_strength{col_suffix}"] = g[f"swing_range{col_suffix}"] / (g["close"] + 1e-9)
        
        # Proyección Fibonacci para TP (Extensión 0.618) y SL (Invalidación de onda)
        g[f"fib_tp{col_suffix}"] = g[f"last_swing_high{col_suffix}"] + (g[f"swing_range{col_suffix}"] * 0.618)
        g[f"fib_sl{col_suffix}"] = g[f"last_swing_low{col_suffix}"] - (g["atr_14"] * 0.5)
        
        # Fase: 1 si está por encima del 61.8% (Impulso), -1 si está por debajo (Corrección)
        threshold_impulse = g[f"last_swing_low{col_suffix}"] + (g[f"swing_range{col_suffix}"] * 0.618)
        g[f"elliott_phase{col_suffix}"] = np.where(g["close"] > threshold_impulse, 1, -1)
        
        # SL/TP dinámicos (Si es corrección, la proyección es más conservadora)
        g[f"structural_sl{col_suffix}"] = np.where(g[f"elliott_phase{col_suffix}"] == 1, g[f"fib_sl{col_suffix}"], g[f"last_swing_high{col_suffix}"] + (g["atr_14"] * 0.5))
        g[f"structural_tp{col_suffix}"] = np.where(g[f"elliott_phase{col_suffix}"] == 1, g[f"fib_tp{col_suffix}"], g[f"last_swing_low{col_suffix}"] - (g[f"swing_range{col_suffix}"] * 0.618))
        
        groups.append(g)
    return pd.concat(groups, ignore_index=True)

def integrate_news_and_market(raw_csv: str, news_file: str, output_parquet: str):
    print("🔄 Iniciando integración avanzada...")
    
    # 1. Convertir CSV a Parquet
    raw_parquet = raw_csv.replace(".csv", ".parquet")
    convert_csv_to_parquet(raw_csv, raw_parquet)
    
    # 2. Cargar y calcular indicadores base
    print("📈 Calculando indicadores técnicos...")
    df_market = pd.read_parquet(raw_parquet)
    df_market["datetime"] = pd.to_datetime(df_market["datetime"], utc=True)
    df_tech = compute_technical_indicators(df_market)
    
    # 3. Aplicar las 3 ventanas de Elliott (60, 80, 100)
    print("📐 Calculando Ondas de Elliott y SL/TP estructurales...")
    df_elliott = apply_elliott_and_levels(df_tech, window=60)
    df_elliott = apply_elliott_and_levels(df_elliott, window=80)
    df_elliott = apply_elliott_and_levels(df_elliott, window=100)
    
    # 4. Dominancia de Elliott (Elegir la ventana con mayor fuerza estructural)
    conditions = [
        (df_elliott["wave_strength_60"] >= df_elliott["wave_strength_80"]) & (df_elliott["wave_strength_60"] >= df_elliott["wave_strength_100"]),
        (df_elliott["wave_strength_80"] >= df_elliott["wave_strength_60"]) & (df_elliott["wave_strength_80"] >= df_elliott["wave_strength_100"]),
        (df_elliott["wave_strength_100"] >= df_elliott["wave_strength_60"]) & (df_elliott["wave_strength_100"] >= df_elliott["wave_strength_80"])
    ]
    choices = [df_elliott["elliott_phase_60"], df_elliott["elliott_phase_80"], df_elliott["elliott_phase_100"]]
    df_elliott["dominant_elliott_phase"] = np.select(conditions, choices, default=0)
    
    # 5. Cruzar con noticias (Opcional según disponibilidad)
    if os.path.exists(news_file):
        print(f"📰 Cruzando con noticias: {news_file}")
        df_news = pd.read_parquet(news_file) if news_file.endswith(".parquet") else pd.read_csv(news_file)
        if "published_at" in df_news.columns:
            df_news["published_at_dt"] = pd.to_datetime(df_news["published_at"], utc=True)
            df_news["date"] = df_news["published_at_dt"].dt.date
            df_elliott["date"] = df_elliott["datetime"].dt.date
            
            # Merge aproximado
            df_final = pd.merge(df_elliott, df_news, how="left", left_on=["date", "ticker"], right_on=["date", "affected_asset"], suffixes=("_market", "_news"))
        else:
            df_final = df_elliott
    else:
        print("⚠️ No hay archivo de noticias. Guardando solo matriz técnica.")
        df_final = df_elliott
        
    # 6. Guardar Parquet maestro
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df_final.to_parquet(output_parquet, engine="pyarrow", compression="snappy", index=False)
    print(f"✅ Dataset maestro guardado en {output_parquet}")

if __name__ == "__main__":
    integrate_news_and_market(
        raw_csv="/home/humberto/financial_data.csv",
        news_file="/home/humberto/news_deepseek.csv",
        output_parquet="/home/humberto/master_ml_dataset.parquet"
      )
