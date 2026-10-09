"""
Pipeline de Ingesta, Filtrado y Análisis de Sentimiento Financiero.
Descarga noticias desde HuggingFace, aplica filtrado previo optimizado con Polars Lazy,
ejecuta inferencia por lotes con FinBERT (20k registros) y agrega las métricas diarias por ticker.
"""

import os
import re
import polars as pl
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm

OUTPUT_PARQUET_PATH = "data/parquet/sentiment_features.parquet"
TARGET_TICKERS = ["AAPL", "NVDA", "TSLA", "GOOGL", "MSFT"]


def extract_ticker(text: str) -> str:
    """Detecta el ticker correspondiente dentro del texto de la noticia."""
    text_upper = text.upper()
    for ticker in TARGET_TICKERS:
        if re.search(r'\b' + ticker + r'\b', text_upper):
            return ticker
    # Reglas secundarias por nombre de compañía
    if "APPLE" in text_upper:
        return "AAPL"
    if "NVIDIA" in text_upper:
        return "NVDA"
    if "TESLA" in text_upper:
        return "TSLA"
    if "GOOGLE" in text_upper or "ALPHABET" in text_upper:
        return "GOOGL"
    if "MICROSOFT" in text_upper:
        return "MSFT"
    return "UNKNOWN"


def run_pipeline():
    os.makedirs("data/parquet", exist_ok=True)
    
    print("1. Cargando conjunto de datos desde HuggingFace (FNSPID / FinGPT)...")
    # Carga del dataset de noticias/sentimiento
    dataset = load_dataset("FinGPT/fingpt-sentiment-train", split="train")
    
    # Conversión eficiente a Polars
    df_raw = pl.from_pandas(dataset.to_pandas())
    print(f"   📊 Registros brutos descargados: {len(df_raw)}")

    # 2. FILTRADO PREVIO ULTRARRÁPIDO CON POLARS LAZY
    print("2. Aplicando filtrado perezoso (Lazy) con Polars para aislar activos objetivo...")
    
    pattern = "(?i)" + "|".join(TARGET_TICKERS + ["Apple", "Nvidia", "Tesla", "Google", "Microsoft"])
    
    df_filtered = (
        df_raw.lazy()
        .filter(pl.col("input").str.contains(pattern))
        .unique(subset=["input"])  # Eliminar duplicados
        .slice(0, 20000)            # Limitar a las 20,000 mejores noticias
        .collect()
    )

    print(f"   🎯 Noticias relevantes filtradas para inferencia: {len(df_filtered)}")

    if len(df_filtered) == 0:
        print("⚠️ No se encontraron noticias coincidentes con los tickers objetivo.")
        return

    # Mapear tickers e identificar fechas
    texts = df_filtered["input"].to_list()
    tickers = [extract_ticker(t) for t in texts]

    # 3. INFERENCIA PARALELIZADA CON FINBERT
    print("3. Cargando modelo FinBERT e iniciando inferencia por lotes (batch_size=64)...")
    
    sentiment_pipeline = pipeline(
        "text-classification",
        model="yiyanghkust/finbert-tone",
        device=-1,       # Uso de CPU en GitHub Codespaces
        batch_size=64    # Paralelización eficiente por lotes
    )

    results = []
    # Inferencia optimizada con barra de progreso
    for out in tqdm(sentiment_pipeline(texts, truncation=True, max_length=128), total=len(texts)):
        results.append(out)

    # Extraer etiquetas y confianzas
    labels = [res["label"] for res in results]
    confidences = [res["score"] for res in results]

    # Mapeo numérico de sentimiento: Positive (+1.0), Neutral (0.0), Negative (-1.0)
    label_map = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}
    sentiment_scores = [label_map.get(lbl, 0.0) * conf for lbl, conf in zip(labels, confidences)]

    # 4. CONSTRUCCIÓN DEL DATAFRAME Y AGREGACIÓN DIARIA EN POLARS
    print("4. Agregando métricas de sentimiento diarias por ticker...")

    # Extraer fecha o asignar fecha por defecto si la columna no existe
    if "date" in df_filtered.columns:
        dates = df_filtered["date"].cast(pl.Utf8).to_list()
    else:
        # En caso de no existir fecha explicita en el subset, mantener estructura de fecha
        dates = ["2024-01-01"] * len(df_filtered)

    df_sentiment = pl.DataFrame({
        "date": dates,
        "ticker": tickers,
        "sentiment_score": sentiment_scores,
        "confidence": confidences
    }).filter(pl.col("ticker") != "UNKNOWN")

    # Agregación diaria por ticker
    df_daily = (
        df_sentiment.lazy()
        .groupby(["date", "ticker"])
        .agg([
            pl.col("sentiment_score").mean().alias("daily_sentiment_score"),
            pl.col("confidence").mean().alias("daily_sentiment_confidence"),
            pl.col("sentiment_score").count().alias("news_volume_count")
        ])
        .sort(["date", "ticker"])
        .collect()
    )

    # Exportar a Parquet
    df_daily.write_parquet(OUTPUT_PARQUET_PATH)
    print(f"✅ Proceso finalizado exitosamente.")
    print(f"💾 Guardado en: {OUTPUT_PARQUET_PATH} ({len(df_daily)} filas agregadas por día/ticker)")


if __name__ == "__main__":
    run_pipeline()
    
