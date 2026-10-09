import polars as pl
from transformers import pipeline

from typing import Dict, Any, List
import polars as pl

"""
Módulo de postprocesamiento e integración de métricas de sentimiento NLP para la capa Silver.
Transforma salidas de FinBERT/Transformers en variables cuantitativas almacenadas en Parquet.
"""

class SentimentPostProcessor:
    """Procesa y agrega los resultados de inferencia NLP (FinBERT) sobre textos financieros."""

    LABEL_MAPPING: Dict[str, float] = {
        "positive": 1.0,
        "neutral": 0.0,
        "negative": -1.0
    }

    def __init__(self, raw_df: pl.DataFrame, inference_results: List[Dict[str, Any]]) -> None:
        """Inicializa el procesador con el DataFrame limpio y los resultados del pipeline.

        Args:
            raw_df (pl.DataFrame): DataFrame de Polars que contiene columnas ['date', 'ticker', 'text'].
            inference_results (List[Dict[str, Any]]): Salida bruta del pipeline de HuggingFace.
        """
        self.raw_df = raw_df
        self.inference_results = inference_results

    def process_and_structure(self) -> pl.DataFrame:
        """Aplica el mapeo numérico, calcula scores ponderados y estructura el DataFrame.

        Returns:
            pl.DataFrame: DataFrame con las columnas de sentimiento individuales.
        """
        # 1. Extraer etiquetas y confianzas de los resultados de HuggingFace
        labels = [res["label"].lower() for res in self.inference_results]
        confidence_scores = [float(res["score"]) for res in self.inference_results]

        # 2. Mapear etiquetas a valores numéricos (-1.0, 0.0, 1.0)
        numeric_values = [self.LABEL_MAPPING.get(lbl, 0.0) for lbl in labels]

        # 3. Añadir columnas vectorizadas al DataFrame
        df_processed = self.raw_df.with_columns([
            pl.Series("sentiment_label", labels, dtype=pl.Utf8),
            pl.Series("sentiment_raw_value", numeric_values, dtype=pl.Float32),
            pl.Series("confidence_score", confidence_scores, dtype=pl.Float32),
        ])

        # 4. Calcular el Score de Sentimiento Ponderado
        df_processed = df_processed.with_columns(
            (pl.col("sentiment_raw_value") * pl.col("confidence_score"))
            .alias("weighted_sentiment_score")
        )

        return df_processed

    def aggregate_daily_sentiment(self, df_processed: pl.DataFrame) -> pl.DataFrame:
        """Agrupa las métricas de sentimiento por fecha y ticker para alinearse con los precios OHLCV.

        Args:
            df_processed (pl.DataFrame): DataFrame con scores individuales por noticia.

        Returns:
            pl.DataFrame: DataFrame agrupado por fecha y ticker listo para la fusión.
        """
        daily_sentiment = df_processed.group_by(["date", "ticker"]).agg([
            pl.col("weighted_sentiment_score").mean().alias("daily_sentiment_mean"),
            pl.col("confidence_score").mean().alias("daily_confidence_mean"),
            pl.col("sentiment_raw_value").count().alias("news_volume_count")
        ]).sort(["ticker", "date"])

        return daily_sentiment


def run_postprocessing_pipeline(
    df_clean: pl.DataFrame, 
    results: List[Dict[str, Any]], 
    output_parquet_path: str = "./parquet/sentiment_features.parquet"
) -> pl.DataFrame:
    """Ejecuta la estructuración, agregación diaria y guardado final en formato Parquet."""
    print("Iniciando postprocesamiento de resultados de sentimiento...")

    processor = SentimentPostProcessor(raw_df=df_clean, inference_results=results)
    
    # Process individual texts
    df_detailed = processor.process_and_structure()
    
    # Aggregate daily for market integration
    df_daily = processor.aggregate_daily_sentiment(df_detailed)

    # Save to Parquet with PyArrow compression
    df_daily.write_parquet(
        output_parquet_path,
        compression="snappy"
    )

    print(f"✅ Guardado final de sentimiento completado con éxito.")
    print(f"Registros agregados por día/ticker: {len(df_daily)}")
    print(f"Archivo generado en: {output_parquet_path}\n")

    return df_daily


if __name__ == "__main__":
    # Lectura eficiente
    df = pl.read_parquet("./parquet/financial_data.parquet")

    # Limpieza inicial
    df_clean = df.filter(
        (pl.col("text").is_not_null()) & 
        (pl.col("text").str.len_bytes() > 10)
    ).unique(subset=["text"])

    #Inferir sobre los textos obtenidos

    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="ProsusAI/finbert", 
        tokenizer="ProsusAI/finbert",
        return_all_scores=False
    )

    # Inferencia sobre los textos
    results = sentiment_pipeline(df_clean["text"].to_list(), batch_size=64)


    df_daily_parquet = run_postprocessing_pipeline(df_clean, results)

      
