"""
Módulo UpdateDailyIncremental (Capa Raw)
----------------------------------------
Descarga de forma incremental los datos más recientes (últimos 5 días en velas de 1 hora)
directamente desde yfinance para la Watchlist, sin añadir ningún indicador ni transformar
el dato virgen, y los concatena con el archivo maestro en la carpeta raw.
"""

import os
from datetime import datetime
from typing import List
import pandas as pd
import yfinance as yf


def update_raw_incremental(
    tickers: List[str] = None,
    parquet_path: str = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/raw/financial_data.parquet"
) -> None:
    """
    Descarga el periodo reciente (5d, 1h) de yfinance y lo concatena de forma limpia 
    con el archivo Parquet raw existente, eliminando duplicados.
    """
    watchlist = tickers if tickers else [
        "IBE.MC",   # Iberdrola S.A.
        "NEE",      # NextEra Energy Inc.
        "GEV",      # GE Vernova LLC
        "CVX",      # Chevron Corporation
        "XOM",      # ExxonMobil Corporation
        "WTI",      # W&T Offshore Inc.
        "BZ=F"      # Brent Crude Oil
    ]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Ejecutando actualización incremental raw (5d, 1h) para {len(watchlist)} activos...")

    new_dfs = []
    failed_tickers = []

    for ticker in watchlist:
        try:
            print(f" 📥 Descargando reciente para {ticker}...")
            # Periodo de 5d e intervalo de 1h según los requerimientos técnicos
            df_ticker = yf.download(
                ticker, 
                period="5d", 
                interval="1h", 
                auto_adjust=True, 
                progress=False
            )
            
            if df_ticker.empty:
                print(f" ⚠️ Advertencia: Sin datos recientes para {ticker}")
                failed_tickers.append(ticker)
                continue

            # Aplanar MultiIndex si es generado por yfinance
            if isinstance(df_ticker.columns, pd.MultiIndex):
                df_ticker.columns = df_ticker.columns.get_level_values(0)

            df_ticker = df_ticker.reset_index()
            df_ticker["ticker"] = ticker
            
            # Normalizar columnas a minúsculas
            df_ticker.columns = [str(col).lower().replace(" ", "_") for col in df_ticker.columns]
            
            new_dfs.append(df_ticker)
            
        except Exception as e:
            print(f" ❌ Error al actualizar {ticker}: {e}")
            failed_tickers.append(ticker)

    if not new_dfs:
        print("❌ No se pudieron obtener nuevos datos en este ciclo incremental.")
        return

    df_new = pd.concat(new_dfs, ignore_index=True)

    # Concatenar con el archivo maestro en la carpeta raw si ya existe
    if os.path.exists(parquet_path):
        print(f" 📂 Leyendo archivo raw existente: {parquet_path}")
        df_existing = pd.read_parquet(parquet_path)
        
        # Unir y eliminar duplicados basándose en datetime y ticker para mantener el dato limpio
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        if "datetime" in df_combined.columns:
            df_combined = df_combined.drop_duplicates(subset=["datetime", "ticker"], keep="last")
            df_combined = df_combined.sort_values(by=["ticker", "datetime"]).reset_index(drop=True)
        else:
            df_combined = df_combined.drop_duplicates().reset_index(drop=True)
    else:
        print(f" 📁 No se encontró archivo previo en raw. Creando el maestro inicial.")
        df_combined = df_new

    # Asegurar directorio de destino
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

    # Guardado optimizado en Parquet con compresión Snappy
    df_combined.to_parquet(
        parquet_path, engine="pyarrow", compression="snappy", index=False
    )

    print("\n" + "="*60)
    print("✅ ACTUALIZACIÓN INCREMENTAL RAW COMPLETADA CON ÉXITO")
    print("="*60)
    print(f" Timestamp de ejecución     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Total filas en archivo raw : {len(df_combined):,}")
    print(f" Ruta del archivo           : {parquet_path}")
    if failed_tickers:
        print(f" Activos con fallo          : {failed_tickers}")
    print("="*60 + "\n")


if __name__ == "__main__":
    update_raw_incremental()
    
