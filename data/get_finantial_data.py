"""
Módulo de Descarga y Procesamiento de Datos Financieros de Mercado.
Descarga datos históricos desde yfinance para el sector energético (XOM, CVX, COP, BP, SHEL),
calcula indicadores técnicos avanzados y genera la variable objetivo (target) en Polars.
"""

import os
import yfinance as yf
import polars as pl
import pandas as pd
import numpy as np

OUTPUT_PARQUET_PATH = "data/parquet/market_data.parquet"
ENERGY_TICKERS = ["XOM", "CVX", "COP", "BP", "SHEL"]
START_DATE = "2018-01-01"


def fetch_historical_data(tickers=ENERGY_TICKERS, start=START_DATE):
    """Descarga datos diarios desde Yahoo Finance para los tickers objetivo."""
    print(f"1. Descargando histórico diario para {tickers} desde {start}...")
    
    df_list = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        df_ticker = stock.history(start=start, interval="1d")
        
        if df_ticker.empty:
            print(f"⚠️ No se obtuvieron datos para {ticker}")
            continue
            
        df_ticker = df_ticker.reset_index()
        df_ticker["ticker"] = ticker
        
        # Estandarizar nombres de columnas
        df_ticker = df_ticker.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        
        # Formatear la fecha a YYYY-MM-DD
        df_ticker["date"] = pd.to_datetime(df_ticker["date"]).dt.strftime("%Y-%m-%d")
        
        cols_to_keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
        df_list.append(df_ticker[cols_to_keep])

    df_combined = pd.concat(df_list, ignore_index=True)
    return pl.from_pandas(df_combined)


def add_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Calcula indicadores técnicos usando expresiones nativas de Polars por cada ticker."""
    print("2. Calculando indicadores técnicos e ingeniería de características...")

    return (
        df.lazy()
        .sort(["ticker", "date"])
        .with_columns([
            # Medias Móviles Exponenciales (EMA)
            pl.col("close").ewm_mean(span=20).over("ticker").alias("ema_20"),
            pl.col("close").ewm_mean(span=50).over("ticker").alias("ema_50"),
            pl.col("close").ewm_mean(span=200).over("ticker").alias("ema_200"),
            
            # Distancia porcentual del precio a las EMAs
            ((pl.col("close") - pl.col("close").ewm_mean(span=20).over("ticker")) / pl.col("close")).alias("dist_ema_20"),
            ((pl.col("close") - pl.col("close").ewm_mean(span=50).over("ticker")) / pl.col("close")).alias("dist_ema_50"),
            ((pl.col("close") - pl.col("close").ewm_mean(span=200).over("ticker")) / pl.col("close")).alias("dist_ema_200"),
            
            # Máximos y Mínimos rodantes a 20 días
            pl.col("high").rolling_max(window_size=20).over("ticker").alias("rolling_high_20"),
            pl.col("low").rolling_min(window_size=20).over("ticker").alias("rolling_low_20"),
        ])
        .with_columns([
            # Indicadores derivados: Rango de precio y Fibonaccis
            (pl.col("rolling_high_20") - pl.col("rolling_low_20")).alias("price_range_20"),
            (pl.col("rolling_low_20") + 0.382 * (pl.col("rolling_high_20") - pl.col("rolling_low_20"))).alias("fibo_382"),
            (pl.col("rolling_low_20") + 0.500 * (pl.col("rolling_high_20") - pl.col("rolling_low_20"))).alias("fibo_500"),
            (pl.col("rolling_low_20") + 0.618 * (pl.col("rolling_high_20") - pl.col("rolling_low_20"))).alias("fibo_618"),
            
            # Oscillator de Elliot (EWO)
            (pl.col("close").ewm_mean(span=5).over("ticker") - pl.col("close").ewm_mean(span=35).over("ticker")).alias("ewo_oscillator"),
        ])
        .with_columns([
            # EWO Ratio
            (pl.col("ewo_oscillator") / pl.col("close")).alias("ewo_ratio"),
            
            # Bandas de Bollinger (20 días, 2 desviaciones estándar)
            pl.col("close").rolling_mean(window_size=20).over("ticker").alias("bb_middle"),
            pl.col("close").rolling_std(window_size=20).over("ticker").alias("bb_std"),
        ])
        .with_columns([
            # %B de Bollinger
            ((pl.col("close") - (pl.col("bb_middle") - 2 * pl.col("bb_std"))) / 
             ((pl.col("bb_middle") + 2 * pl.col("bb_std")) - (pl.col("bb_middle") - 2 * pl.col("bb_std")))).alias("bb_percent"),
            
            # MACD (12, 26, 9)
            (pl.col("close").ewm_mean(span=12).over("ticker") - pl.col("close").ewm_mean(span=26).over("ticker")).alias("macd_line"),
        ])
        .with_columns([
            pl.col("macd_line").ewm_mean(span=9).over("ticker").alias("macd_signal"),
        ])
        .with_columns([
            (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_hist"),
            
            # Target de Swing Trading: Máximo profit potencial alcanzado en los próximos 10 días
            pl.col("high").shift(-10).rolling_max(window_size=10).over("ticker").alias("forward_max_high")
        ])
        .with_columns([
            # Retorno porcentual máximo a futuro
            ((pl.col("forward_max_high") - pl.col("close")) / pl.col("close")).alias("forward_max_profit")
        ])
        .with_columns([
            # Variable binaria objetivo: 1 si el profit potencial es >= 4% en la ventana de 10 días
            (pl.col("forward_max_profit") >= 0.04).cast(pl.Int32).alias("target_high_profit")
        ])
        .drop(["bb_middle", "bb_std"])
        .collect()
    )


def run_get_data():
    os.makedirs("data/parquet", exist_ok=True)
    
    df_raw = fetch_historical_data()
    df_processed = add_technical_indicators(df_raw)
    
    # Limpiar valores nulos por ventanas rodantes
    df_final = df_processed.drop_nulls()
    
    df_final.write_parquet(OUTPUT_PARQUET_PATH)
    print(f"✅ Proceso completado exitosamente.")
    print(f"💾 Guardado en: {OUTPUT_PARQUET_PATH} ({len(df_final)} filas procesadas)")


if __name__ == "__main__":
    run_get_data()
