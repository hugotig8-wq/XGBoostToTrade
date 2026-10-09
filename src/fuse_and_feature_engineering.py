"""
Módulo de Fusión y Feature Engineering Avanzado con Target de Profit Futuro (Elliott Wave Strategy).
Compatible con Polars 2.0+ (.str.to_date) y con soporte completo para Ondas de Elliott y Fibonacci.
"""

import os
import polars as pl


class FinancialFeatureEngineer:
    def __init__(
        self,
        prices_parquet_path: str = "/workspaces/XGBoostToTrade/data/parquet/market_data.parquet",
        sentiment_parquet_path: str = "/workspaces/XGBoostToTrade/data/parquet/sentiment_features.parquet"
    ) -> None:
        self.prices_path = prices_parquet_path
        self.sentiment_path = sentiment_parquet_path

    def load_and_fuse_data(self) -> pl.DataFrame:
        print("1. Cargando y fusionando datos de mercado con sentimiento de noticias...")
        
        # Corrección de Polars 2.0: Conversión explícita usando .str.to_date()
        df_prices = pl.read_parquet(self.prices_path).with_columns(
            pl.col("date").str.to_date("%Y-%m-%d")
        )
        df_sentiment = pl.read_parquet(self.sentiment_path).with_columns(
            pl.col("date").str.to_date("%Y-%m-%d")
        )

        df_fused = df_prices.join(
            df_sentiment,
            on=["date", "ticker"],
            how="left"
        ).with_columns([
            pl.col("daily_sentiment_score").fill_null(0.0),
            pl.col("daily_sentiment_confidence").fill_null(0.0),
            pl.col("news_volume_count").fill_null(0)
        ]).sort(["ticker", "date"])

        return df_fused

    def compute_elliott_wave_oscillator(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calcula el Oscilador de Ondas de Elliott (EWO = SMA(5) - SMA(35))
        y su ratio sobre el precio de cierre para identificar la Onda 3 y 5.
        """
        print("2. Calculando Oscilador de Ondas de Elliott (EWO)...")
        df_ewo = df.with_columns([
            (
                pl.col("close").rolling_mean(window_size=5).over("ticker") - 
                pl.col("close").rolling_mean(window_size=35).over("ticker")
            ).alias("ewo_oscillator")
        ]).with_columns([
            (pl.col("ewo_oscillator") / pl.col("close")).alias("ewo_ratio")
        ])
        return df_ewo

    def compute_fibonacci_features(self, df: pl.DataFrame, lookback_window: int = 20) -> pl.DataFrame:
        """
        Extrae el rango de precio a 20 días para determinar zonas clave de retroceso
        de Fibonacci (38.2%, 50.0% y 61.8%) y posición dentro del impulso.
        """
        print(f"3. Calculando niveles de Retroceso de Fibonacci (Ventana: {lookback_window}d)...")
        df_fibo = df.with_columns([
            pl.col("high").rolling_max(window_size=lookback_window).over("ticker").alias("rolling_high_20"),
            pl.col("low").rolling_min(window_size=lookback_window).over("ticker").alias("rolling_low_20")
        ]).with_columns([
            (pl.col("rolling_high_20") - pl.col("rolling_low_20")).alias("price_range_20")
        ]).with_columns([
            (pl.col("rolling_high_20") - (pl.col("price_range_20") * 0.382)).alias("fibo_382"),
            (pl.col("rolling_high_20") - (pl.col("price_range_20") * 0.500)).alias("fibo_500"),
            (pl.col("rolling_high_20") - (pl.col("price_range_20") * 0.618)).alias("fibo_618")
        ])
        return df_fibo

    def compute_forward_profit_target(
        self, 
        df: pl.DataFrame, 
        forward_horizon: int = 10, 
        profit_threshold: float = 0.04
    ) -> pl.DataFrame:
        """
        Calcula el Máximo Profit Futuro dentro de un horizonte de H días hacia adelante.
        
        - forward_max_high: El precio máximo alcanzado en los próximos H días (t+1 a t+H).
        - forward_max_profit: % máximo de rendimiento alcanzable desde el Cierre t.
        - target_high_profit: 1 si el máximo profit es >= threshold (ej. 4%), 0 de lo contrario.
        """
        print(f"4. Calculando Forward Max Profit (Horizonte futuro: {forward_horizon}d)...")
        
        # Invertimos temporalmente el dataframe por ticker para calcular máximos hacia adelante
        df_target = df.reverse().with_columns([
            pl.col("high").shift(1).rolling_max(window_size=forward_horizon).over("ticker").alias("forward_max_high")
        ]).reverse()

        # Calculamos el % de ganancia máxima y la etiqueta binaria
        df_target = df_target.with_columns([
            ((pl.col("forward_max_high") - pl.col("close")) / pl.col("close")).alias("forward_max_profit"),
            pl.col("daily_sentiment_score").rolling_mean(window_size=3).over("ticker").alias("sentiment_ma_3"),
            pl.col("daily_sentiment_score").rolling_mean(window_size=7).over("ticker").alias("sentiment_ma_7")
        ]).with_columns([
            (pl.col("forward_max_profit") >= profit_threshold).cast(pl.Int32).alias("target_high_profit")
        ]).drop_nulls()

        return df_target


def main() -> None:
    engineer = FinancialFeatureEngineer()
    df_fused = engineer.load_and_fuse_data()
    df_ewo = engineer.compute_elliott_wave_oscillator(df_fused)
    df_fibo = engineer.compute_fibonacci_features(df_ewo, lookback_window=20)
    
    # Horizonte futuro de 10 días hábiles con meta de profit del 4%
    df_dataset = engineer.compute_forward_profit_target(df_fibo, forward_horizon=10, profit_threshold=0.04)

    output_path = "/workspaces/XGBoostToTrade/data/parquet/ml_ready_dataset_enhanced.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_dataset.write_parquet(output_path, compression="snappy")

    print("\n" + "="*65)
    print("✅ Refactorización Completada Exitosamente.")
    print(f"Filas totales procesadas : {len(df_dataset)}")
    print(f"Horizonte Futuro Evaluado: 10 Días Hábiles")
    print(f"Archivo actualizado en   : {output_path}")
    print("="*65)
    print("\nMuestra con Target de Profit Futuro:")
    print(df_dataset.select(["date", "ticker", "close", "forward_max_high", "forward_max_profit", "target_high_profit"]).head(10))


if __name__ == "__main__":
    main()
    
