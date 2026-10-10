"""
Módulo de Ingeniería de Características Técnicas.
Calcula indicadores de tendencia, momento y volatilidad usando Polars.
"""

import polars as pl
import numpy as np


def add_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Añade indicadores técnicos estandarizados y relativos al DataFrame de precios."""
    
    # Asegurar orden temporal por ticker y fecha
    df = df.sort(["ticker", "date"])

    # 1. EMAs Relativas (Porcentaje de distancia del precio respecto a la media)
    # Expresar como (Close - EMA) / EMA evita problemas de escala con el tiempo
    for span in [20, 50, 200]:
        df = df.with_columns(
            pl.col("close")
            .ewm_mean(span=span, adjust=False)
            .over("ticker")
            .alias(f"ema_{span}")
        )
        df = df.with_columns(
            ((pl.col("close") - pl.col(f"ema_{span}")) / pl.col(f"ema_{span}"))
            .alias(f"dist_ema_{span}")
        )

    # 2. RSI (Relative Strength Index - 14 períodos)
    df = df.with_columns(
        (pl.col("close") - pl.col("close").shift(1).over("ticker")).alias("change")
    )
    df = df.with_columns([
        pl.when(pl.col("change") > 0).then(pl.col("change")).otherwise(0).alias("gain"),
        pl.when(pl.col("change") < 0).then(-pl.col("change")).otherwise(0).alias("loss")
    ])
    df = df.with_columns([
        pl.col("gain").ewm_mean(span=14, adjust=False).over("ticker").alias("avg_gain"),
        pl.col("loss").ewm_mean(span=14, adjust=False).over("ticker").alias("avg_loss")
    ])
    df = df.with_columns(
        (100 - (100 / (1 + (pl.col("avg_gain") / (pl.col("avg_loss") + 1e-9))))).alias("rsi_14")
    )

    # 3. MACD (12, 26, 9) Normalizado por el Precio
    df = df.with_columns([
        pl.col("close").ewm_mean(span=12, adjust=False).over("ticker").alias("ema_12"),
        pl.col("close").ewm_mean(span=26, adjust=False).over("ticker").alias("ema_26")
    ])
    df = df.with_columns(
        ((pl.col("ema_12") - pl.col("ema_26")) / pl.col("close")).alias("macd_line")
    )
    df = df.with_columns(
        pl.col("macd_line").ewm_mean(span=9, adjust=False).over("ticker").alias("macd_signal")
    )
    df = df.with_columns(
        (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_hist")
    )

    # 4. Bandas de Bollinger (20 períodos, 2 desviaciones estándar)
    df = df.with_columns([
        pl.col("close").rolling_mean(window_size=20).over("ticker").alias("bb_middle"),
        pl.col("close").rolling_std(window_size=20).over("ticker").alias("bb_std")
    ])
    df = df.with_columns([
        (pl.col("bb_middle") + (2 * pl.col("bb_std"))).alias("bb_upper"),
        (pl.col("bb_middle") - (2 * pl.col("bb_std"))).alias("bb_lower")
    ])
    # Posición % dentro de las bandas (0 = Banda inferior, 1 = Banda superior)
    df = df.with_columns(
        ((pl.col("close") - pl.col("bb_lower")) / (pl.col("bb_upper") - pl.col("bb_lower") + 1e-9)).alias("bb_percent")
    )

    # Limpieza de columnas intermedias
    drop_cols = [
        "change", "gain", "loss", "avg_gain", "avg_loss",
        "ema_12", "ema_26", "ema_20", "ema_50", "ema_200",
        "bb_middle", "bb_std", "bb_upper", "bb_lower"
    ]
    df = df.drop([c for c in drop_cols if c in df.columns])

    # Eliminar nulos generados por ventanas dinámicas
    return df.drop_nulls()


if __name__ == "__main__":
    # Prueba del módulo con el dataset procesado
    df_input = pl.read_parquet("/workspaces/XGBoostToTrade/ColectorAgent/data/parquet/ml_ready_dataset.parquet")
    df_enhanced = add_technical_indicators(df_input)
    
    # Guardar versión extendida
    df_enhanced.write_parquet("/workspaces/XGBoostToTrade/ColectorAgent/data/parquet/ml_ready_dataset_enhanced.parquet")
    
    print(f"✅ Features calculadas exitosamente. Nuevas columnas creadas:")
    print([c for c in df_enhanced.columns if c not in df_input.columns])
  
