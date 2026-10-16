"""
Módulo ColectorRawAgent (Datos Vírgenes Directos)
------------------------------------------------
Descarga los datos históricos y diarios de la Watchlist directamente de yfinance
sin agregar ningún índice técnico ni transformación, y los guarda en la capa raw.
"""

import os
from datetime import datetime
from typing import List
import pandas as pd
import yfinance as yf


def fetch_and_save_raw_market_data(
    tickers: List[str], 
    output_dir: str = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/raw"
) -> None:
    """Descarga datos crudos (Raw) desde yfinance y los guarda directamente en Parquet."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Descargando datos vírgenes para {len(tickers)} activos...")
    
    processed_dfs = []
    failed_tickers = []

    for ticker in tickers:
        try:
            print(f" 📥 Descargando activo crudo: {ticker}...")
            # Descarga de histórico amplio en escala diaria (o 60d para intradiario si se requiere)
            df_ticker = yf.download(
                ticker, period="5y", interval="1d", auto_adjust=True, progress=False
            )
            
            if df_ticker.empty:
                print(f" ⚠️ Advertencia: No se obtuvieron datos para {ticker}")
                failed_tickers.append(ticker)
                continue

            # Aplanar MultiIndex si es generado por yfinance
            if isinstance(df_ticker.columns, pd.MultiIndex):
                df_ticker.columns = df_ticker.columns.get_level_values(0)

            df_ticker = df_ticker.reset_index()
            df_ticker["ticker"] = ticker
            
            # Normalizar cabeceras a minúsculas para estandarizar el esquema
            df_ticker.columns = [str(col).lower().replace(" ", "_") for col in df_ticker.columns]
            
            processed_dfs.append(df_ticker)
            
        except Exception as e:
            print(f" ❌ Error al descargar {ticker}: {e}")
            failed_tickers.append(ticker)

    if not processed_dfs:
        print("❌ No se pudo procesar ningún activo.")
        return

    final_df = pd.concat(processed_dfs, ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, "financial_data.parquet")

    # Guardado directo en formato Parquet comprimido con Snappy
    final_df.to_parquet(
        parquet_path, engine="pyarrow", compression="snappy", index=False
    )

    print("\n" + "="*60)
    print("✅ RESGUARDO DE DATOS VÍRGENES EN CAPA RAW COMPLETADO")
    print("="*60)
    print(f" Timestamp de ejecución     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Filas totales almacenadas  : {len(final_df):,}")
    print(f" Archivo guardado en        : {parquet_path}")
    if failed_tickers:
        print(f" Activos fallidos           : {failed_tickers}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Watchlist oficial de activos energéticos y materias primas
    watchlist_tickers = [
        "IBE.MC",   # Iberdrola S.A.
        "NEE",      # NextEra Energy Inc.
        "GEV",      # GE Vernova LLC
        "CVX",      # Chevron Corporation
        "XOM",      # ExxonMobil Corporation
        "WTI",      # W&T Offshore Inc.
        "BZ=F"      # Brent Crude Oil (EuroOIL)
    ]
    
    fetch_and_save_raw_market_data(tickers=watchlist_tickers)
    
