"""
Módulo de Fusión de Datos y Generación de Indicadores Financieros Avanzados.
Combina precios bursátiles con sentiment_features.parquet y calcula
el Oscilador de Ondas de Elliott (EWO) y niveles de Fibonacci usando Polars.
"""

import os
import polars as pl


class FinancialFeatureEngineer:
    """Clase encargada de fusionar datasets y construir variables para XGBoost."""

    def __init__(
        self,
        prices_parquet_path: str = "../data/parquet/financial_data.parquet",
        sentiment_parquet_path: str = "../data/data/parquet/sentiment_features.parquet"
    ) -> None:
        self.prices_path = prices_parquet_path
        self.sentiment_path = sentiment_parquet_path

    def load_and_fuse_data(self) -> pl.DataFrame:
        """Carga ambos archivos Parquet y realiza un Left Join por date y ticker."""
        print("1. Cargando datasets de precios y sentimiento...")
        df_prices = pl.read_parquet(self.prices_path)
        df_sentiment = pl.read_parquet(self.sentiment_path)

        # Asegurar tipo Date en ambas tablas
        df_prices = df_prices.with_columns(pl.col("date").cast(pl.Date))
        df_sentiment = df_sentiment.with_columns(pl.col("date").cast(pl.Date))

        print("2. Uniendo precios con métricas de sentimiento...")
        df_fused = df_prices.join(
            df_sentiment,
            on=["date", "ticker"],
            how="left"
        ).with_columns([
            # Imputación de días sin noticias
            pl.col("daily_sentiment_score").fill_null(0.0),
            pl.col("daily_sentiment_confidence").fill_null(0.0),
            pl.col("news_volume_count").fill_null(0)
        ]).sort(["ticker", "date"])

        return df_fused

    def compute_elliott_wave_oscillator(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calcula el Oscilador de Ondas de Elliott (EWO = SMA_5 - SMA_35)."""
        print("3. Calculando el Oscilador de Ondas de Elliott (EWO)...")
        df_ewo = df.with_columns([
            pl.col("close").apply_many(
                lambda s: s, return_dtype=pl.Float64
            ) if False else 
            (
                pl.col("close").rolling_mean(window_size=5).over("ticker") - 
                pl.col("close").rolling_mean(window_size=35).over("ticker")
            ).alias("ewo_oscillator")
        ])
        
        # Ratio EWO normalizado por el precio de cierre
        df_ewo = df_ewo.with_columns(
            (pl.col("ewo_oscillator") / pl.col("close")).alias("ewo_ratio")
        )
        return df_ewo

    def compute_fibonacci_features(self, df: pl.DataFrame, window_size: int = 20) -> pl.DataFrame:
        """Calcula los niveles de retroceso de Fibonacci móvil de 20 días."""
        print(f"4. Calculando niveles de retroceso de Fibonacci (Ventana: {window_size}d)...")
        
        df_fibo = df.with_columns([
            pl.col("high").rolling_max(window_size=window_size).over("ticker").alias("rolling_high_20"),
            pl.col("low").rolling_min(window_size=window_size).over("ticker").alias("rolling_low_20")
        ]).with_columns([
            (pl.col("rolling_high_20") - pl.col("rolling_low_20")).alias("price_range_20")
        ]).with_columns([
            (pl.col("rolling_high_20") - (pl.col("price_range_20") * 0.382)).alias("fibo_382"),
            (pl.col("rolling_high_20") - (pl.col("price_range_20") * 0.500)).alias("fibo_500"),
            (pl.col("rolling_high_20") - (pl.col("price_range_20") * 0.618)).alias("fibo_618")
        ])
        return df_fibo

    def compute_target_and_lags(self, df: pl.DataFrame) -> pl.DataFrame:
        """Genera la variable objetivo (Market Direction: 1=Alza, 0=Baja) y Lags."""
        print("5. Generando variable objetivo (Target Direction) y Lags de sentimiento...")
        
        df_final = df.with_columns([
            # Target: Retorno a 1 día en el futuro (Dirección Alza/Baja)
            (pl.col("close").shift(-1).over("ticker") > pl.col("close"))
            .cast(pl.Int32)
            .alias("target_direction"),
            
            # Promedios móviles de sentimiento a 3 y 7 días
            pl.col("daily_sentiment_score").rolling_mean(window_size=3).over("ticker").alias("sentiment_ma_3"),
            pl.col("daily_sentiment_score").rolling_mean(window_size=7).over("ticker").alias("sentiment_ma_7")
        ]).dropna()  # Elimina filas iniciales/finales generadas por ventanas flotantes
        
        return df_final


def main() -> None:
    engineer = FinancialFeatureEngineer()
    df_fused = engineer.load_and_fuse_data()
    df_ewo = engineer.compute_elliott_wave_oscillator(df_fused)
    df_fibo = engineer.compute_fibonacci_features(df_ewo)
    df_dataset = engineer.compute_target_and_lags(df_fibo)

    output_path = "../data/parquet/ml_ready_dataset.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_dataset.write_parquet(output_path, compression="snappy")

    print("\n" + "="*60)
    print("✅ Dataset Final de Machine Learning Generado Correctamente.")
    print(f"Filas totales listos para XGBoost : {len(df_dataset)}")
    print(f"Columnas creadas                  : {len(df_dataset.columns)}")
    print(f"Archivo guardado en               : {output_path}")
    print("="*60)
    print("\nMuestra del dataset combinado:")
    print(df_dataset.select(["date", "ticker", "close", "daily_sentiment_score", "ewo_oscillator", "fibo_500", "target_direction"]).head(10))


if __name__ == "__main__":
    main()
      
