"""
run_refinement.py
=================
Pipeline de refinamiento para news_deepseek4.csv.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from ing_avanzada_features import (
    build_news_features,
    cluster_news,
    collapse_clusters,
    normalize_news,
)

SCHEMA_ALIASES = {
    "timestamp":  ["timestamp", "published_at", "published", "date", "datetime", "created_at"],
    "title":      ["title", "headline", "headlines", "head"],
    "abstract":   ["abstract", "summary", "body", "text", "content", "description", "lead"],
    "sentiment":  ["sentiment_raw", "sentiment_score", "sentiment", "score"],
    "confidence": ["confidence_raw", "confidence_score", "confidence", "conf"],
    "impact":     ["impact_raw", "impact_magnitude", "impact_score", "impact"],
    "assets":     ["asset", "assets", "tickers", "symbols", "ticker", "affected_asset"],
    "actors":     ["actors_raw", "actors", "entities", "organizations", "event_actor"],
    "topics":     ["topics_raw", "topics", "themes", "categories", "sectors", "event_type"],
    "source":     ["source", "publisher", "domain"],
}

def resolve_columns(df: pd.DataFrame) -> dict:
    resolved = {}
    for key, candidates in SCHEMA_ALIASES.items():
        for c in candidates:
            if c in df.columns:
                resolved[key] = c
                break
        else:
            resolved[key] = None
    return resolved

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", default="news_deepseek4.csv") # <-- Nombre por defecto actualizado
    ap.add_argument("--bars", default=None)
    ap.add_argument("--mode", choices=["news-only", "merge"], default="news-only")
    ap.add_argument("--collapse", choices=["none", "max_conf", "mean"], default="none")
    ap.add_argument("--half-life-h", type=float, default=12.0)
    ap.add_argument("--max-age-h", type=float, default=120.0)
    ap.add_argument("--window-cluster-h", type=float, default=6.0)
    ap.add_argument("--sim-threshold", type=float, default=0.85)
    ap.add_argument("--out-news", default="news_enriched.csv")
    ap.add_argument("--out-bars", default="bars_features_v2.parquet")
    args = ap.parse_args()

    news = pd.read_csv(args.news)
    print(f"[news] {len(news)} filas, {len(news.columns)} columnas")

    cols = resolve_columns(news)
    print("[schema] resuelto:")
    for k, v in cols.items():
        print(f"  {k:<12} → {v}")

    # 1. Renombrar columnas base (las numéricas y de tiempo)
    news = news.rename(columns={
        cols["timestamp"]: "timestamp",
        cols["sentiment"]: "sentiment_raw",
        cols["confidence"]: "confidence_raw",
        cols["impact"]: "impact_raw",
        cols["assets"]: "asset",
    })

    # 2. Renombrar textos
    if cols["title"]: news = news.rename(columns={cols["title"]: "title"})
    if cols["abstract"]: news = news.rename(columns={cols["abstract"]: "abstract"})

    # 3. Combinar actores y topics (basado en el CSV de la captura anterior)
    if "event_actor" in news.columns and "subject_entity" in news.columns:
        news["actors_raw"] = news["event_actor"].astype(str) + ", " + news["subject_entity"].astype(str)
    if "event_type" in news.columns and "sector" in news.columns:
        news["topics_raw"] = news["event_type"].astype(str) + ", " + news["sector"].astype(str)

    # 4. Limpieza de Asset
    news["asset"] = news["asset"].astype(str).str.strip()
    news = news[news["asset"] != "nan"]

    # 5. Normalizar
    news = normalize_news(
        news,
        sentiment_col="sentiment_raw",
        confidence_col="confidence_raw",
        impact_col="impact_raw",
    )
    print(f"[news] normalizado: {len(news)} filas, {news['asset'].nunique()} assets únicos.")

    # 6. Clustering (usará abstract automáticamente porque lo pusimos en text_cols)
    news = cluster_news(
        news,
        time_window_h=args.window_cluster_h,
        sim_threshold=args.sim_threshold,
    )
    n_clusters = news["cluster_id"].nunique()
    print(f"[cluster] {n_clusters} clusters, tamaño máx: {news['cluster_size'].max()}")

    # 7. Colapsar
    if args.collapse != "none":
        news_eff = collapse_clusters(news, method=args.collapse)
        print(f"[collapse/{args.collapse}] {len(news_eff)} filas efectivas.")
    else:
        news_eff = news

    # 8. Guardar noticias
    news_eff.to_csv(args.out_news, index=False)
    print(f"[save] {args.out_news}")

    # 9. Merge con barras
    if args.mode == "merge":
        if not args.bars:
            print("[ERROR] --bars requerido en modo merge")
            sys.exit(1)
        
        bars_path = Path(args.bars)
        bars = pd.read_parquet(bars_path) if bars_path.suffix == ".parquet" else pd.read_csv(bars_path)
        
        feats = build_news_features(
            news_eff, bars, windows=(3, 5),
            half_life_h=args.half_life_h, max_age_h=args.max_age_h
        )
        feats.to_parquet(args.out_bars, index=False)
        print(f"[save] {args.out_bars}")

if __name__ == "__main__":
    main()
