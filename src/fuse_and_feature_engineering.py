"""
Módulo de Fusión y Feature Engineering para QML.
Integra precios históricos, análisis de sentimiento de cripto/energía,
ondas de Elliott y retrocesos de Fibonacci de forma compatible con Polars moderno.
"""

import os
import polars as pl

MARKET_DATA_PATH = "/workspace/XGBoostToTrade/ColectorAgent/data/parquet/market_historical_data.parquet"
SENTIMENT_DATA_PATH = "/workspace/XGBoostToTrade/ColectorAgent/data/parquet/crypto_sentiment_features.parquet"
OUTPUT_DATASET_PATH = "/workspace/XGBoostToTrade/ColectorAgent/data/parquet/ml_ready_dataset_enhanced.parquet"


def fuse_and_engineer_features() -> None:
    """Ejecuta la fusión de fuentes y calcula indicadores técnicos avanzados."""
    print("1. Verificando existencia de archivos Parquet...")
    
    if not os.path.exists(MARKET_DATA_PATH) or not os.path.exists(SENTIMENT_DATA_PATH):
        raise FileNotFoundError(
            "⚠️ Asegúrate de haber ejecutado los scripts de descarga de mercado y sentimiento previamente."
        )

    df_market = pl.read_parquet(MARKET_DATA_PATH)
    df_sentiment = pl.read_parquet(SENTIMENT_DATA_PATH)

    print("2. Normalizando esquemas y fechas de forma segura...")
    
    # Conversión robusta de fecha soportando tanto String como Date previo
    df_market = df_market.with_columns(
        pl.when(pl.col("date").is_not_null())
        .then(pl.col("date").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False))
        .otherwise(None)
        .alias("date")
    )
    
    df_sentiment = df_sentiment.with_columns(
        pl.when(pl.col("date").is_not_null())
        .then(pl.col("date").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False))
        .otherwise(None)
        .alias("date")
    )

    print("3. Uniendo datos de mercado con el sentimiento de criptoactivos y energía...")
    
    df_fused = (
        df_market.lazy()
        .join(df_sentiment.lazy(), on=["date", "ticker"], how="left")
        .sort(["ticker", "date"])
        .with_columns([
            pl.col("daily_sentiment_score").fill_null(0.0),
            pl.col("daily_sentiment_confidence").fill_null(0.0),
            pl.col("news_volume_count").fill_null(0),
        ])
    )

    print("4. Calculando Ondas de Elliott y Retrocesos de Fibonacci...")
    
    # Simulación de Ondas de Elliott y Niveles Fibonacci sobre la ventana móvil de precios
    df_engineered = (
        df_fused
        .with_columns([
            # Medias móviles base para estructura de ondas
            pl.col("close").rolling_mean(window_size=5).over("ticker").alias("ma_wave_5"),
            pl.col("close").rolling_mean(window_size=20).over("ticker").alias("ma_wave_20"),
        ])
        .with_columns([
            # Diferencial de Elliott Wave Oscillator (EWO) aproximado
            (pl.col("ma_wave_5") - pl.col("ma_wave_20")).alias("elliott_wave_oscillator"),
            
            # Cálculo dinámico de soportes/resistencias estilo Fibonacci (38.2% y 61.8%)
            (pl.col("close").rolling_max(window_size=30).over("ticker")).alias("rolling_high_30"),
            (pl.col("close").rolling_min(window_size=30).over("ticker")).alias("rolling_low_30"),
        ])
        .with_columns([
            # Niveles de Fibonacci aplicados al rango local
            (pl.col("rolling_high_30") - (pl.col("rolling_high_30") - pl.col("rolling_low_30")) * 0.382).alias("fib_level_382"),
            (pl.col("rolling_high_30") - (pl.col("rolling_high_30") - pl.col("rolling_low_30")) * 0.618).alias("fib_level_618"),
        ])
        .drop_nulls()
        .collect()
    )

    os.makedirs(os.path.dirname(OUTPUT_DATASET_PATH), exist_ok=True)
    df_engineered.write_parquet(OUTPUT_DATASET_PATH, compression="snappy")

    print("\n" + "="*60)
    print("✅ Fusión e Ingeniería de Características completada con éxito.")
    print(f"💾 Dataset guardado en : {OUTPUT_DATASET_PATH}")
    print(f"📊 Registros totales   : {len(df_engineered)}")
    print("="*60)


if __name__ == "__main__":
    fuse_and_engineer_features()
    
