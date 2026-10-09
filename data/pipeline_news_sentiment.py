"""
Módulo de Integración de Noticias Reales e Inferencia de Sentimiento con FinBERT.
Descarga datasets masivos de noticias financieras, calcula scores de polaridad
y genera variables agregadas diarias en formato Parquet optimizado.
"""

import os
from typing import Dict, List, Any, Optional
import polars as pl
import torch
from transformers import pipeline


class FinancialNewsFetcher:
    """Encargado de la descarga y filtrado de noticias históricas reales."""

    FNSPID_URL: str = (
        "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/"
        "Stock_news/nasdaq_external_data.csv"
    )

    def __init__(self, target_tickers: List[str]) -> None:
        """Inicializa el extractor con los tickers de interés.

        Args:
            target_tickers (List[str]): Lista de tickers (ej. ['AAPL', 'NVDA']).
        """
        self.target_tickers = target_tickers

    def fetch_filtered_news(self, max_records_per_ticker: Optional[int] = 5000) -> pl.DataFrame:
        """Lee el dataset masivo desde HuggingFace mediante Polars LazyFrame (Streaming).

        Args:
            max_records_per_ticker (Optional[int]): Límite de noticias por ticker.

        Returns:
            pl.DataFrame: DataFrame con noticias reales filtradas.
        """
        print("1. Iniciando descarga/filtrado de noticias reales desde HuggingFace (FNSPID)...")

        try:
            # Uso de scan_csv para procesamiento perezoso (sin cargar todo a la RAM)
            lazy_df = pl.scan_csv(self.FNSPID_URL, ignore_errors=True)

            # Filtrar por los tickers requeridos y seleccionar columnas relevantes
            df_filtered = (
                lazy_df
                .filter(pl.col("Ticker").is_in(self.target_tickers))
                .select([
                    pl.col("Date").alias("raw_date"),
                    pl.col("Ticker").alias("ticker"),
                    pl.col("Article_title").alias("text")
                ])
                .collect(engine="streaming")
            )

            if df_filtered.is_empty():
                print("⚠️ No se encontraron noticias para los tickers especificados.")
                return pl.DataFrame()

            # Normalización y cast de fechas
            df_cleaned = (
                df_filtered
                .with_columns(
                    pl.col("raw_date")
                    .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
                    .dt.date()
                    .alias("date")
                )
                .drop("raw_date")
                .filter(pl.col("text").is_not_null() & (pl.col("text").str.len_bytes() > 10))
                .unique(subset=["date", "ticker", "text"])
            )

            # Muestreo controlado si excede el límite
            if max_records_per_ticker:
                df_cleaned = (
                    df_cleaned
                    .group_by("ticker")
                    .head(max_records_per_ticker)
                )

            print(f"✅ Noticias reales filtradas: {len(df_cleaned)} titulares listos.")
            return df_cleaned

        except Exception as e:
            print(f"❌ Error al descargar o parsear noticias: {e}")
            return pl.DataFrame()


class FinBERTSentimientoEngine:
    """Inferencia de sentimiento financiero mediante el Transformer ProsusAI/finbert."""

    LABEL_MAPPING: Dict[str, float] = {
        "positive": 1.0,
        "neutral": 0.0,
        "negative": -1.0
    }

    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 32) -> None:
        """Inicializa el modelo de HuggingFace en GPU o CPU.

        Args:
            model_name (str): Identificador del modelo preentrenado.
            batch_size (int): Tamaño del lote para inferencia en paralelo.
        """
        self.batch_size = batch_size
        self.device = 0 if torch.cuda.is_available() else -1
        
        print(f"2. Cargando modelo FinBERT ('{model_name}') en dispositivo: {'GPU' if self.device == 0 else 'CPU'}...")
        self.pipe = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=self.device,
            return_all_scores=False
        )

    def analyze_texts(self, df_news: pl.DataFrame) -> pl.DataFrame:
        """Ejecuta inferencia por lotes sobre la columna de texto.

        Args:
            df_news (pl.DataFrame): DataFrame con columna 'text'.

        Returns:
            pl.DataFrame: DataFrame enriquecido con predicciones individuales.
        """
        texts = df_news["text"].to_list()
        print(f"3. Iniciando inferencia FinBERT sobre {len(texts)} titulares (Batch Size: {self.batch_size})...")

        # Inferencia optimizada en lotes
        raw_outputs: List[Dict[str, Any]] = self.pipe(texts, batch_size=self.batch_size)

        # Mapeo numérico y extracción
        labels = [res["label"].lower() for res in raw_outputs]
        confidences = [float(res["score"]) for res in raw_outputs]
        raw_values = [self.LABEL_MAPPING.get(lbl, 0.0) for lbl in labels]

        # Creación de columnas de sentimiento
        df_analyzed = df_news.with_columns([
            pl.Series("sentiment_label", labels, dtype=pl.Utf8),
            pl.Series("confidence_score", confidences, dtype=pl.Float32),
            pl.Series("sentiment_raw", raw_values, dtype=pl.Float32)
        ]).with_columns(
            (pl.col("sentiment_raw") * pl.col("confidence_score")).alias("weighted_sentiment")
        )

        return df_analyzed


class DailySentimentAggregator:
    """Agrega métricas de sentimiento a nivel de día y ticker para alinear con OHLCV."""

    @staticmethod
    def aggregate_and_save(
        df_analyzed: pl.DataFrame,
        output_parquet_path: str = "data/parquet/sentiment_features.parquet"
    ) -> pl.DataFrame:
        """Agrupa los resultados por fecha y ticker y exporta a Parquet.

        Args:
            df_analyzed (pl.DataFrame): DataFrame con inferencias individuales.
            output_parquet_path (str): Ruta de destino del archivo Parquet.

        Returns:
            pl.DataFrame: DataFrame agregado por día.
        """
        print("4. Agregando métricas de sentimiento diarias...")

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

        # Crear directorio y guardar Parquet
        os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
        df_daily.write_parquet(output_parquet_path, compression="snappy")

        print("\n" + "="*60)
        print("✅ Pipeline de Noticias y Sentimiento finalizado con éxito.")
        print(f"Total de días/tickers registrados : {len(df_daily)}")
        print(f"Archivo Parquet resultante       : {output_parquet_path}")
        print("="*60)

        return df_daily


def main() -> None:
    """Ejecución principal del pipeline."""
    # Tickers de interés para el proyecto de inversión
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]

    # 1. Extracción de noticias reales
    fetcher = FinancialNewsFetcher(target_tickers=tickers)
    df_news = fetcher.fetch_filtered_news(max_records_per_ticker=2000)

    if not df_news.is_empty():
        # 2. Inferencia NLP FinBERT
        engine = FinBERTSentimientoEngine(batch_size=32)
        df_analyzed = engine.analyze_texts(df_news)

        # 3. Postprocesamiento y exportación Parquet
        df_daily_features = DailySentimentAggregator.aggregate_and_save(df_analyzed)

        print("\nMuestra del dataset final de sentimiento (Parquet):")
        print(df_daily_features.head(10))


if __name__ == "__main__":
    main()
        
