"""
Pipeline de Sentimiento NATIVO Crypto.
Descarga noticias masivas sobre Bitcoin, Ethereum y Altcoins desde HuggingFace,
aplica filtrado perezoso con Polars Lazy y procesa con ProsusAI/finbert.
"""

import os
import re
import polars as pl
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm

OUTPUT_PARQUET_PATH = "data/parquet/crypto_sentiment_features.parquet"

# Mapeo de activos crypto y términos asociados
TARGET_ASSETS = {
    "BTC": ["BITCOIN", "BTC"],
    "ETH": ["ETHEREUM", "ETHER", "ETH"],
    "SOL": ["SOLANA", "SOL"],
    "XRP": ["RIPPLE", "XRP"],
    "ADA": ["CARDANO", "ADA"],
    "DOGE": ["DOGECOIN", "DOGE"]
}

MAX_NEWS_PER_ASSET = 10000


def extract_crypto_asset(text: str) -> str:
    """Identifica la criptomoneda en el texto."""
    if not text:
        return "UNKNOWN"
    text_upper = text.upper()
    for symbol, keywords in TARGET_ASSETS.items():
        for kw in keywords:
            if re.search(r'\b' + kw + r'\b', text_upper):
                return symbol
    return "UNKNOWN"


def run_pipeline():
    os.makedirs("data/parquet", exist_ok=True)
    
    print("1. Cargando conjunto de datos NATIVO DE CRYPTO desde HuggingFace...")
    # Dataset masivo dedicado a noticias de Criptomonedas
    try:
        dataset = load_dataset("ElKev/crypto-news-sentiment", split="train")
    except Exception:
        # Source secundario masivo de crypto
        dataset = load_dataset("m-a-p/crypto_news", split="train")
    
    df_raw = pl.from_pandas(dataset.to_pandas())
    print(f"   📊 Registros brutos descargados: {len(df_raw)}")

    # Identificar columna de texto automáticamente
    text_col = None
    for col in ["text", "title", "headline", "input", "news"]:
        if col in df_raw.columns:
            text_col = col
            break

    if not text_col:
        text_col = df_raw.columns[0]

    # 2. MAPEO Y FILTRADO LAZY CON POLARS
    print("2. Procesando y clasificando noticias por criptoactivo con Polars...")
    
    texts_raw = df_raw[text_col].cast(pl.Utf8).fill_null("").to_list()
    asset_labels = [extract_crypto_asset(t) for t in texts_raw]

    df_mapped = df_raw.with_columns([
        pl.Series("input_text", texts_raw),
        pl.Series("ticker", asset_labels)
    ]).filter(pl.col("ticker") != "UNKNOWN")

    # Limitar a máximo 10,000 por activo
    df_filtered = (
        df_mapped.lazy()
        .unique(subset=["input_text"])
        .group_by("ticker")
        .head(MAX_NEWS_PER_ASSET)
        .sort("ticker")
        .collect()
    )

    print(f"   🎯 Total de noticias crypto relevantes encontradas: {len(df_filtered)}")
    counts = df_filtered.group_by("ticker").len()
    for row in counts.iter_rows():
        print(f"      - {row[0]}: {row[1]} noticias")

    if len(df_filtered) == 0:
        print("⚠️ No se encontraron suficientes coincidencia de activos en el dataset.")
        return

    texts = df_filtered["input_text"].to_list()
    tickers = df_filtered["ticker"].to_list()

    # 3. INFERENCIA CON PROSUSAI/FINBERT
    print("3. Ejecutando inferencia de sentimiento con ProsusAI/finbert (batch_size=64)...")
    
    sentiment_pipeline = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        device=-1,
        batch_size=64
    )

    results = []
    for out in tqdm(sentiment_pipeline(texts, truncation=True, max_length=128), total=len(texts)):
        results.append(out)

    labels = [res["label"] for res in results]
    confidences = [res["score"] for res in results]

    label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    sentiment_scores = [label_map.get(lbl.lower(), 0.0) * conf for lbl, conf in zip(labels, confidences)]

    # 4. AGREGACIÓN Y EXPORTACIÓN A PARQUET
    print("4. Agregando métricas diarias por criptoactivo...")

    # Capturar columna de fecha si existe
    date_col = None
    for col in ["date", "created_at", "time", "timestamp"]:
        if col in df_filtered.columns:
            date_col = col
            break

    if date_col:
        dates = df_filtered[date_col].cast(pl.Utf8).str.slice(0, 10).to_list()
    else:
        dates = ["2024-01-01"] * len(df_filtered)

    df_sentiment = pl.DataFrame({
        "date": dates,
        "ticker": tickers,
        "sentiment_score": sentiment_scores,
        "confidence": confidences
    })

    df_daily = (
        df_sentiment.lazy()
        .group_by(["date", "ticker"])
        .agg([
            pl.col("sentiment_score").mean().alias("daily_sentiment_score"),
            pl.col("confidence").mean().alias("daily_sentiment_confidence"),
            pl.col("sentiment_score").count().alias("news_volume_count")
        ])
        .sort(["date", "ticker"])
        .collect()
    )

    df_daily.write_parquet(OUTPUT_PARQUET_PATH)
    print(f"✅ Inferencia finalizada con éxito.")
    print(f"💾 Guardado en: {OUTPUT_PARQUET_PATH} ({len(df_daily)} filas agregadas)")


if __name__ == "__main__":
    run_pipeline()
    
