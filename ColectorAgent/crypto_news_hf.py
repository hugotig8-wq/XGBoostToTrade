"""
Módulo de Ingesta y Análisis de Sentimiento para Criptoactivos.
Filtra noticias de Hugging Face, procesa la polaridad con FinBERT y 
genera métricas agregadas diarias en formato Parquet optimizado.
"""

import os
from typing import Dict, List, Any, Optional
import polars as pl
import torch
from transformers import pipeline


class CryptoNewsFetcher:
    """Encargado de la descarga y filtrado de noticias masivas de criptomonedas."""

    # Dataset abierto de referencia para noticias y sentimiento cripto/financiero
    CRYPTO_NEWS_URL: str = (
        "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/"
        "Stock_news/nasdaq_exteral_data.csv"
    )

    def __init__(self, target_assets: List[str]) -> None:
        """Inicializa el extractor para los criptoactivos seleccionados.

        Args:
            target_assets (List[str]): Lista de criptoactivos (ej. ['BTC', 'ETH', 'SOL', 'XRP']).
        """
        self.target_assets = target_assets

    def fetch_filtered_crypto_news(self, max_records_per_asset: int = 10000) -> pl.DataFrame:
        """Filtra noticias específicas de criptoactivos usando Polars LazyFrame (Streaming).

        Args:
            max_records_per_asset (int): Límite máximo estricto de noticias por criptoactivo.

        Returns:
            pl.DataFrame: DataFrame limpio con noticias de criptomonedas.
        """
        print("1. Iniciando descarga y filtrado de noticias de criptoactivos desde HuggingFace...")

        try:
            lazy_df = pl.scan_csv(self.CRYPTO_NEWS_URL, ignore_errors=True)

            # Filtrar por los símbolos de criptoactivos definidos
            df_filtered = (
                lazy_df
                .filter(pl.col("Stock_symbol").is_in(self.target_assets))
                .select([
                    pl.col("Date").alias("raw_date"),
                    pl.col("Stock_symbol").alias("ticker"),
                    pl.col("Article_title").alias("text")
                ])
                .collect(engine="streaming")
            )

            if df_filtered.is_empty():
                print("⚠️ No se encontraron registros para los criptoactivos indicados.")
                return pl.DataFrame()

            # Normalización estricta de fechas y limpieza de textos vacíos
            df_cleaned = (
                df_filtered
                .with_columns(
                    pl.col("raw_date")
                    .str.slice(0, 10)
                    .str.to_date("%Y-%m-%d", strict=False)
                    .alias("date")
                )
                .drop("raw_date")
                .filter(
                    pl.col("date").is_not_null() & 
                    pl.col("text").is_not_null() & 
                    (pl.col("text").str.len_bytes() > 10)
                )
                .unique(subset=["date", "ticker", "text"])
            )

            # Aplicar límite estricto de hasta 10,000 registros por criptoactivo
            if max_records_per_asset:
                df_cleaned = (
                    df_cleaned
                    .group_by("ticker")
                    .head(max_records_per_asset)
                )

            print(f"✅ Criptonoticias filtradas: {len(df_cleaned)} registros listos.")
            return df_cleaned

        except Exception as e:
            print(f"❌ Error en la ingesta de criptonoticias: {e}")
            return pl.DataFrame()


class CryptoSentimentEngine:
    """Inferencia de sentimiento orientada a criptoactivos mediante FinBERT."""

    LABEL_MAPPING: Dict[str, float] = {
        "positive": 1.0,
        "neutral": 0.0,
        "negative": -1.0
    }

    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 32) -> None:
        """Inicializa el modelo transformers para análisis de polaridad.

        Args:
            model_name (str): Identificador del modelo en Hugging Face.
            batch_size (int): Lote de procesamiento paralelo en hardware.
        """
        self.batch_size = batch_size
        self.device = 0 if torch.cuda.is_available() else -1
        
        print(f"2. Cargando FinBERT ('{model_name}') en dispositivo: {'GPU' if self.device == 0 else 'CPU'}...")
        self.pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=self.device,
            return_all_scores=False
        )

    def analyze_crypto_texts(self, df_news: pl.DataFrame) -> pl.DataFrame:
        """Ejecuta inferencia por lotes sobre los titulares de criptomonedas.

        Args:
            df_news (pl.DataFrame): DataFrame con columna 'text'.

        Returns:
            pl.DataFrame: DataFrame enriquecido con scores de confianza y polaridad.
        """
        texts = df_news["text"].to_list()
        print(f"3. Procesando inferencia de sentimiento en {len(texts)} titulares cripto...")

        raw_outputs: List[Dict[str, Any]] = self.pipe(texts, batch_size=self.batch_size)

        labels = [res["label"].lower() for res in raw_outputs]
        confidences = [float(res["score"]) for res in raw_outputs]
        raw_values = [self.LABEL_MAPPING.get(lbl, 0.0) for lbl in labels]

        df_analyzed = df_news.with_columns([
            pl.Series("sentiment_label", labels, dtype=pl.Utf8),
            pl.Series("confidence_score", confidences, dtype=pl.Float32),
            pl.Series("sentiment_raw", raw_values, dtype=pl.Float32)
        ]).with_columns(
            (pl.col("sentiment_raw") * pl.col("confidence_score")).alias("weighted_sentiment")
        )

        return df_analyzed


class CryptoSentimentAggregator:
    """Agrega métricas diarias de sentimiento para correlacionarlas con precios OHLCV."""

    @staticmethod
    def aggregate_and_save(
        df_analyzed: pl.DataFrame,
        output_parquet_path: str = "data/parquet/crypto_sentiment_features.parquet"
    ) -> pl.DataFrame:
        """Agrupa las valoraciones por fecha y criptoactivo y exporta a Parquet.

        Args:
            df_analyzed (pl.DataFrame): DataFrame con inferencias individuales.
            output_parquet_path (str): Ruta de destino del archivo Parquet.

        Returns:
            pl.DataFrame: DataFrame consolidado.
        """
        print("4. Agregando métricas de sentimiento diarias para criptomonedas...")

        df_daily = (
            df_analyzed
            .group_by(["date", "ticker"])
            .agg([
                pl.col("weighted_sentiment").mean().alias("daily_sentiment_score"),
                pl.col("confidence_score").mean().alias("daily_sentiment_confidence"),
                pl.col("text").count().alias("news_volume_count")
            ])
            .sort(["ticker", "date"])
        )

        os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
        df_daily.write_parquet(output_parquet_path, compression="snappy")

        print("\n" + "="*60)
        print("✅ Pipeline Crypto de Sentimiento finalizado con éxito.")
        print(f"Total de registros consolidados : {len(df_daily)}")
        print(f"Archivo Parquet guardado en     : {output_parquet_path}")
        print("="*60)

        return df_daily


def main() -> None:
    """Ejecución principal del pipeline enfocado en criptoactivos."""
    crypto_assets = ["BTC", "ETH", "SOL", "XRP"]

    fetcher = CryptoNewsFetcher(target_assets=crypto_assets)
    df_news = fetcher.fetch_filtered_crypto_news(max_records_per_asset=10000)

    if not df_news.is_empty():
        engine = CryptoSentimentEngine(batch_size=32)
        df_analyzed = engine.analyze_crypto_texts(df_news)

        df_daily_features = CryptoSentimentAggregator.aggregate_and_save(df_analyzed)

        print("\nMuestra del dataset cripto final (Parquet):")
        print(df_daily_features.head(10))


if __name__ == "__main__":
    main()
      
