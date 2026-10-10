"""
Módulo de Descarga y Procesamiento de Datos Históricos (OHLCV).
Conecta con yfinance para extraer la evolución de precios de criptoactivos y energía,
almacenando los resultados de forma optimizada en formato Parquet.
"""

import os
from typing import List
import pandas as pd
import yfinance as yf


def fetch_and_save_financial_data(
    tickers: List[str], 
    period: str = "5y", 
    output_dir: str = "data/parquet"
) -> None:
    """Descarga de forma segura datos históricos desde yfinance y los consolida en Parquet.

    Args:
        tickers (List[str]): Lista de símbolos o pares (ej. ['BTC-USD', 'ETH-USD', 'CL=F']).
        period (str): Rango histórico de extracción (ej. '5y').
        output_dir (str): Directorio local para almacenamiento Parquet.
    """
    print(f"Iniciando descarga histórica para {len(tickers)} activos (período: {period})...")
    
    processed_dfs = []
    failed_tickers = []

    for ticker in tickers:
        try:
            print(f"Descargando activo: {ticker}...")
            df_ticker = yf.download(
                ticker, period=period, interval="1d", auto_adjust=True, progress=False
            )
            
            if df_ticker.empty:
                print(f"⚠️ Advertencia: No se obtuvieron datos para {ticker}")
                failed_tickers.append(ticker)
                continue

            # Aplanar MultiIndex si es generado por yfinance
            if isinstance(df_ticker.columns, pd.MultiIndex):
                df_ticker.columns = df_ticker.columns.get_level_values(0)

            df_ticker = df_ticker.reset_index()
            df_ticker["ticker"] = ticker
            
            # Normalizar cabeceras a minúsculas
            df_ticker.columns = [str(col).lower().replace(" ", "_") for col in df_ticker.columns]
            
            processed_dfs.append(df_ticker)
            
        except Exception as e:
            print(f"❌ Error al descargar {ticker}: {e}")
            failed_tickers.append(ticker)

    if not processed_dfs:
        print("No se pudo procesar ningún activo financiero.")
        return

    final_df = pd.concat(processed_dfs, ignore_index=True)

    # Normalización de fechas y particionamiento temporal
    final_df["date"] = pd.to_datetime(final_df["date"]).dt.date
    final_df["year"] = pd.to_datetime(final_df["date"]).dt.year
    final_df["month"] = pd.to_datetime(final_df["date"]).dt.month

    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, "market_historical_data.parquet")

    final_df.to_parquet(
        parquet_path, engine="pyarrow", compression="snappy", index=False
    )

    print("\n" + "="*50)
    print("✅ Ingesta de datos de mercado completada con éxito.")
    print(f"Filas totales almacenadas : {len(final_df)}")
    print(f"Archivo guardado en      : {parquet_path}")
    if failed_tickers:
        print(f"Activos con error        : {failed_tickers}")
    print("="*50)


if __name__ == "__main__":
    # Cartera combinada: Criptoactivos principales y Futuros de Energía
    target_portfolio = [
        "BTC-USD",  # Bitcoin
        "ETH-USD",  # Ethereum
        "SOL-USD",  # Solana
        "XRP-USD",  # Ripple
        "CL=F"      # Petróleo Crudo WTI (Energía)
    ]
    fetch_and_save_financial_data(tickers=target_portfolio, period="5y")
  
