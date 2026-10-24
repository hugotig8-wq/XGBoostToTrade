"""
news_inspect.py
===============
Diagnóstico rápido del CSV de noticias. No modifica nada.
Uso:
    python news_inspect.py news_deepseek3.csv
"""

import sys
import json
import pandas as pd
import numpy as np


def _show_list_col(df, col, n=10):
    if col not in df.columns:
        return
    print(f"\n--- {col} (top {n}) ---")
    counts = {}
    for v in df[col].dropna():
        items = []
        if isinstance(v, list):
            items = v
        elif isinstance(v, str):
            try:
                j = json.loads(v)
                items = j if isinstance(j, list) else [j]
            except Exception:
                items = [t.strip() for t in v.split(",") if t.strip()]
        for it in items:
            it = str(it).strip().lower()
            if it:
                counts[it] = counts.get(it, 0) + 1
    for k, v in sorted(counts.items(), key=lambda x: -x[1])[:n]:
        print(f"  {k:<40} {v}")


def main(path):
    df = pd.read_csv(path)
    print(f"Archivo: {path}")
    print(f"Shape: {df.shape}")
    print(f"\nColumnas ({len(df.columns)}):")
    for c in df.columns:
        print(f"  - {c}  ({df[c].dtype})")

    print("\nPrimeras 5 filas:")
    print(df.head().to_string())

    # Tiempo
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower():
            try:
                parsed = pd.to_datetime(df[c], utc=True, errors="coerce")
                if parsed.notna().sum() > 0:
                    print(f"\n{c}: {parsed.min()} → {parsed.max()}  "
                          f"(nulos: {parsed.isna().sum()})")
            except Exception:
                pass

    # Numéricas
    print("\nEstadísticas numéricas:")
    num = df.select_dtypes(include=[np.number])
    if len(num.columns):
        print(num.describe().T.to_string())

    # Listas / categóricas
    for c in df.columns:
        if c.lower() in ("actors", "entities", "organizations",
                         "assets", "tickers", "symbols",
                         "topics", "themes", "categories", "sectors"):
            _show_list_col(df, c)

    # Duplicados por título
    for tcol in ("title", "headline"):
        if tcol in df.columns:
            dup = df[df.duplicated(tcol, keep=False)]
            print(f"\nDuplicados exactos por '{tcol}': {len(dup)}")
            if len(dup):
                print(dup[[tcol]].head(10).to_string())
            break

    # Filas vacías de texto
    for tcol in ("title", "summary", "body", "text"):
        if tcol in df.columns:
            empty = df[tcol].isna() | (df[tcol].astype(str).str.strip() == "")
            print(f"\nFilas vacías en '{tcol}': {empty.sum()}/{len(df)}")
            break


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "news_deepseek3.csv"
    main(path)
