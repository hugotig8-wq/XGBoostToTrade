"""
Pipeline de Ingesta, Filtrado y Análisis de Sentimiento Financiero para Crypto.
Descarga noticias/comentarios desde HuggingFace, filtra por pares de criptoactivos,
limita a un máximo de 10,000 noticias por activo con Polars Lazy y aplica FinBERT (batch_size=64).
"""

import os
import re
import polars as pl
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm

OUTPUT_PARQUET_PATH = "data/parquet/sentiment_features.parquet"

# Simbolos/Tickers objetivos y sus nombres comunes
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
    """Detecta el criptoactivo correspondiente dentro del texto de la noticia."""
    text_upper = text.upper()
    
    for symbol, keywords in TARGET_ASSETS.items():
        for kw in keywords:
            # Coincidencia por palabra completa para evitar falsos positivos
            if re.search(r'\b' + kw + r'\b', text_upper):
                return symbol
    return "UNKNOWN"


def run_pipeline():
    os.makedirs("data/parquet", exist_ok=True)
    
    print("1. Cargando conjunto de datos Cripto/Financiero desde HuggingFace...")
    # Se utiliza el dataset de sentimientos financieros
    dataset = load_dataset("FinGPT/fingpt-sentiment-train", split="train")
    
    df_raw = pl.from_pandas(dataset.to_pandas())
    print(f"   📊 Registros brutos descargados: {len(df_raw)}")

    # 2. FILTRADO PREVIO Y LIMITACIÓN A 10,000 NOTICIAS POR ACTIVO CON POLARS
    print("2. Aplicando filtrado perezoso (Lazy) con Polars y límites por criptoactivo...")
    
    # Construir expresiones de búsqueda regex
    all_keywords = [kw for keywords in TARGET_ASSETS.values() for kw in keywords]
    pattern = "(?i)" + "|".join(all_keywords)
    
    # Mapear activo primero para aplicar el límite por grupo de forma exacta
    texts_raw = df_raw["input"].to_list()
    asset_labels = [extract_crypto_asset(t) for t in texts_raw]

    df_mapped = df_raw.with_columns(
        pl.Series("ticker", asset_labels)
    ).filter(pl.col("ticker") != "UNKNOWN")

    # Aplicar deduplicación y tomar máximo 10,000 noticias por cada activo
    df_filtered = (
        df_mapped.lazy()
        .unique(subset=["input"])
        .group_by("ticker")
        .head(MAX_NEWS_PER_ASSET)
        .sort("ticker")
        .collect()
    )

    print(f"   🎯 Total de noticias filtradas: {len(df_filtered)}")
    
    # Imprimir conteo por criptoactivo
    counts = df_filtered.group_by("ticker").len()
    for row in counts.iter_rows():
        print(f"      - {row[0]}: {row[1]} noticias")

    if len(df_filtered) == 0:
        print("⚠️ No se encontraron noticias coincidentes con los criptoactivos objetivo.")
        return

    texts = df_filtered["input"].to_list()
    tickers = df_filtered["ticker"].to_list()

    # 3. INFERENCIA PARALELIZADA CON FINBERT (BATCH_SIZE=64)
    print("3. Cargando modelo FinBERT e iniciando inferencia por lotes (batch_size=64)...")
    
    sentiment_pipeline = pipeline(
        "text-classification",
        model="yiyanghkust/finbert-tone",
        device=-1,       # Uso de CPU en GitHub Codespaces
        batch_size=64    # Inferencia por lotes paralelizada
    )

    results = []
    for out in tqdm(sentiment_pipeline(texts, truncation=True, max_length=128), total=len(texts)):
        results.append(out)

    labels = [res["label"] for res in results]
    confidences = [res["score"] for res in results]

    # Mapeo numérico: Positive (+1.0), Neutral (0.0), Negative (-1.0)
    label_map = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}
    sentiment_scores = [label_map.get(lbl, 0.0) * conf for lbl, conf in zip(labels, confidences)]

    # 4. CONSTRUCCIÓN DEL DATAFRAME Y AGREGACIÓN DIARIA EN POLARS
    print("4. Agregando métricas diarias de sentimiento por criptoactivo...")

    if "date" in df_filtered.columns:
        dates = df_filtered["date"].cast(pl.Utf8).to_list()
    else:
        dates = ["2024-01-01"] * len(df_filtered)

    df_sentiment = pl.DataFrame({
        "date": dates,
        "ticker": tickers,
        "sentiment_score": sentiment_scores,
        "confidence": confidences
    })

    # Agregación diaria
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

    # Exportar resultado
    df_daily.write_parquet(OUTPUT_PARQUET_PATH)
    print(f"✅ Pipeline Crypto finalizado con éxito.")
    print(f"💾 Guardado en: {OUTPUT_PARQUET_PATH} ({len(df_daily)} filas agregadas por día/activo)")


if __name__ == "__main__":
    run_pipeline()
  
