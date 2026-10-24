"""
run_refinement.py
=================
Pipeline:
  1. Cargar `news_deepseek3.csv`.
  2. Resolver nombres de columnas.
  3. Normalizar (timestamps, clip, impacto saturado).
  4. Clustering por actor/tema/embedding/asset.
  5. (Opcional) colapsar clusters a representantes.
  6. Agregar a barras con ventanas N=3, N=5.
  7. Guardar `news_enriched.csv` y `bars_features_v2.parquet`.

Ejemplos:
  python run_refinement.py --mode news-only
  python run_refinement.py --mode merge --bars bars.parquet
  python run_refinement.py --mode merge --bars bars.parquet --collapse max_conf
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

# ---------------------------------------------------------------------------
# Mapeo de nombres: cada clave del esquema interno puede venir con varios alias.
# ---------------------------------------------------------------------------
SCHEMA_ALIASES = {
    "timestamp":  ["timestamp", "published_at", "published", "date", "datetime", "created_at"],
    "title":      ["title", "headline", "headlines", "head"],
    "summary":    ["summary", "body", "text", "content", "description", "lead"],
    "sentiment":  ["sentiment_raw", "sentiment_score", "sentiment", "score"],
    "confidence": ["confidence_raw", "confidence_score", "confidence", "conf"],
    "impact":     ["impact_raw", "impact_magnitude", "impact_score", "impact"],
    "assets":     ["asset", "assets", "tickers", "symbols", "ticker"],
    "actors":     ["actors_raw", "actors", "entities", "organizations"],
    "topics":     ["topics_raw", "topics", "themes", "categories", "sectors"],
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


def explode_assets(news: pd.DataFrame, assets_col: str) -> pd.DataFrame:
    """
    Si `assets` es una lista o string con varios assets, expandir a una fila
    por (noticia, asset). Si ya hay una fila por asset, no toca nada.
    """
    sample = news[assets_col].dropna().iloc[0] if len(news) else None
    needs_explosion = False
    if isinstance(sample, list) and len(sample) > 1:
        needs_explosion = True
    elif isinstance(sample, str):
        try:
            j = json.loads(sample)
            if isinstance(j, list) and len(j) > 1:
                needs_explosion = True
        except Exception:
            if "," in sample:
                needs_explosion = True
    if not needs_explosion:
        return news

    rows = []
    for _, row in news.iterrows():
        v = row[assets_col]
        if isinstance(v, list):
            assets = v
        elif isinstance(v, str):
            try:
                j = json.loads(v)
                assets = j if isinstance(j, list) else [v]
            except Exception:
                assets = [t.strip() for t in v.split(",") if t.strip()]
        else:
            continue
        for a in assets:
            r = row.copy()
            r["asset"] = str(a).strip()
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", default="news_deepseek3.csv")
    ap.add_argument("--bars", default=None, help="parquet/csv con las 53 features")
    ap.add_argument("--mode", choices=["news-only", "merge"], default="news-only")
    ap.add_argument("--collapse", choices=["none", "max_conf", "mean"], default="none")
    ap.add_argument("--half-life-h", type=float, default=12.0)
    ap.add_argument("--max-age-h", type=float, default=120.0)
    ap.add_argument("--window-cluster-h", type=float, default=6.0)
    ap.add_argument("--sim-threshold", type=float, default=0.85)
    ap.add_argument("--out-news", default="news_enriched.csv")
    ap.add_argument("--out-bars", default="bars_features_v2.parquet")
    args = ap.parse_args()

    # 1) Cargar noticias
    news = pd.read_csv(args.news)
    print(f"[news] {len(news)} filas, {len(news.columns)} columnas")

    # 2) Resolver esquema
    cols = resolve_columns(news)
    print("[schema] resuelto:")
    for k, v in cols.items():
        print(f"  {k:<12} → {v}")

    required = ["timestamp", "sentiment", "confidence", "impact", "assets"]
    missing = [k for k in required if cols[k] is None]
    if missing:
        print(f"[ERROR] faltan columnas requeridas: {missing}")
        print("Renombra tu CSV o ajusta SCHEMA_ALIASES.")
        sys.exit(1)

    # 3) Canonicalizar nombres mínimos
    news = news.rename(columns={
        cols["timestamp"]: "timestamp",
        cols["sentiment"]: "sentiment_raw",
        cols["confidence"]: "confidence_raw",
        cols["impact"]: "impact_raw",
        cols["assets"]: "asset",
    })
    if cols["actors"]:
        news = news.rename(columns={cols["actors"]: "actors_raw"})
    if cols["topics"]:
        news = news.rename(columns={cols["topics"]: "topics_raw"})

    # 4) Explosión si hay varios assets por noticia
    news = explode_assets(news, "asset")
    news["asset"] = news["asset"].astype(str).str.strip()

    # 5) Normalizar
    news = normalize_news(
        news,
        sentiment_col="sentiment_raw",
        confidence_col="confidence_raw",
        impact_col="impact_raw",
    )
    print(f"[news] tras normalizar: {len(news)} filas, "
          f"{news['asset'].nunique()} assets, "
          f"rango {news['timestamp'].min()} → {news['timestamp'].max()}")

    # 6) Clustering
    news = cluster_news(
        news,
        time_window_h=args.window_cluster_h,
        sim_threshold=args.sim_threshold,
    )
    n_clusters = news["cluster_id"].nunique()
    print(f"[cluster] {n_clusters} clusters, "
          f"tamaño medio {news['cluster_size'].mean():.2f}, "
          f"max {news['cluster_size'].max()}")

    # 7) Colapsar (opcional)
    if args.collapse != "none":
        news_eff = collapse_clusters(news, method=args.collapse)
        print(f"[collapse/{args.collapse}] {len(news_eff)} filas efectivas")
    else:
        news_eff = news

    # 8) Guardar noticias enriquecidas
    news_eff.to_csv(args.out_news, index=False)
    print(f"[save] {args.out_news}")

    # 9) Modo merge: agregar a barras
    if args.mode == "merge":
        if not args.bars:
            print("[ERROR] --bars requerido en modo merge")
            sys.exit(1)
        bars_path = Path(args.bars)
        if bars_path.suffix == ".parquet":
            bars = pd.read_parquet(bars_path)
        else:
            bars = pd.read_csv(bars_path)

        print(f"[bars] {len(bars)} filas, {len(bars.columns)} columnas")
        feats = build_news_features(
            news_eff,
            bars,
            windows=(3, 5),
            half_life_h=args.half_life_h,
            max_age_h=args.max_age_h,
        )
        print(f"[features] {len(feats)} filas, {len(feats.columns)} columnas "
              f"({len(feats.columns) - len(bars.columns)} nuevas)")
        feats.to_parquet(args.out_bars, index=False)
        print(f"[save] {args.out_bars}")


if __name__ == "__main__":
    main()
