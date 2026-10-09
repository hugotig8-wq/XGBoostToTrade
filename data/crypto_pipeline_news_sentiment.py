"""
Pipeline de Sentimiento para el Sector Energético (Petróleo y Gas).
Filtra noticias de XOM, CVX, COP, BP, SHEL desde el dataset masivo de HuggingFace,
aplica un límite de 10,000 registros por activo con Polars Lazy y procesa con FinBERT (batch_size=64).
"""

import os
import re
import polars as pl
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm

OUTPUT_PARQUET_PATH = "/workspaces/XGBoostToTrade/data/parquet/sentiment_features.parquet"

# Mapeo de empresas energéticas principales y sus términos
TARGET_ENERGY = {
    "XOM": ["EXXON", "EXXONMOBIL", "XOM"],
    "CVX": ["CHEVRON", "CVX"],
    "COP": ["CONOCOPHILLIPS", "COP"],
    "BP": ["BP", "BRITISH PETROLEUM"],
    "SHEL": ["SHELL", "ROYAL DUTCH SHELL", "SHEL"]
}

MAX_NEWS_PER_ASSET = 10000


def extract_energy_ticker(text: str) -> str:
    """Detecta la empresa energética correspondiente dentro del texto de la noticia."""
    if not text:
        return "UNKNOWN"
    text_upper = str(text).upper()
    for symbol, keywords in TARGET_ENERGY.items():
        for kw in keywords:
            if re.search(r'\b' + kw + r'\b', text_upper):
                return symbol
    return "UNKNOWN"


def run_pipeline():
    os.makedirs("data/parquet", exist_ok=True)
    
    print("1. Cargando conjunto de datos financiero desde HuggingFace (FinGPT / FNSPID)...")
    dataset = load_dataset("FinGPT/fingpt-sentiment-train", split="train")
    
    df_raw = pl.from_pandas(dataset.to_pandas())
    print(f"   📊 Registros brutos descargados: {len(df_raw)}")

    # Detectar columna de texto
    text_col = "input" if "input" in df_raw.columns else df_raw.columns[0]

    # 2. FILTRADO PREVIO Y MAPEO CON POLARS LAZY
    print("2. Filtrando y mapeando noticias del sector energético con Polars...")
    
    texts_raw = df_raw[text_col].cast(pl.Utf8).fill_null("").to_list()
    ticker_labels = [extract_energy_ticker(t) for t in texts_raw]

    df_mapped = df_raw.with_columns([
        pl.Series("input_text", texts_raw),
        pl.Series("ticker", ticker_labels)
    ]).filter(pl.col("ticker") != "UNKNOWN")

    # Limitar a máximo 10,000 noticias por cada activo energético
    df_filtered = (
        df_mapped.lazy()
        .unique(subset=["input_text"])
        .group_by("ticker")
        .head(MAX_NEWS_PER_ASSET)
        .sort("ticker")
        .collect()
    )

    print(f"   🎯 Total de noticias de energía filtradas: {len(df_filtered)}")
    counts = df_filtered.group_by("ticker").len()
    for row in counts.iter_rows():
        print(f"      - {row[0]}: {row[1]} noticias")

    if len(df_filtered) == 0:
        print("⚠️ No se encontraron noticias de empresas energéticas.")
        return

    texts = df_filtered["input_text"].to_list()
    tickers = df_filtered["ticker"].to_list()

    # 3. INFERENCIA CON PROSUSAI/FINBERT
    print("3. Cargando modelo ProsusAI/finbert e iniciando inferencia (batch_size=64)...")
    
    sentiment_pipeline = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        device=-1,       # Uso de CPU
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
    print("4. Agregando métricas diarias por activo...")

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
    print(f"✅ Proceso completado exitosamente.")
    print(f"💾 Guardado en: {OUTPUT_PARQUET_PATH}")


if __name__ == "__main__":
    run_pipeline()
  
