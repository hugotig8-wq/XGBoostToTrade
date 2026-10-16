"""
Módulo de Actualización Diaria Incremental
------------------------------------------
Descarga únicamente el último día de cotización para los activos de la Watchlist
y lo concatena con el archivo Parquet histórico existente, evitando duplicados.
"""

import os
import pandas as pd
import yfinance as yf

def update_latest_market_day(
    tickers: list[str], 
    parquet_path: str = "data/parquet/financial_data.parquet"
) -> None:
    """Descarga el último día disponible de yfinance y lo fusiona con el histórico."""
    print("🔄 Consultando el último día de mercado para la Watchlist...")
    
    new_data_frames = []
    for ticker in tickers:
        try:
            # period='5d' asegura capturar el último día hábil incluso tras fines de semana o festivos
            df = yf.download(ticker, period="5d", interval="1d", auto_adjust=True, progress=False)
            if df.empty:
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            df = df.reset_index()
            df["ticker"] = ticker
            df.columns = [str(col).lower().replace(" ", "_") for col in df.columns]
            
            # Normalizar fecha a formato fecha pura
            df["date"] = pd.to_datetime(df["date"]).dt.date
            
            # Seleccionar estrictamente la fecha más reciente disponible
            latest_date = df["date"].max()
            df_latest = df[df["date"] == latest_date].copy()
            new_data_frames.append(df_latest)
            
        except Exception as e:
            print(f"❌ Error al obtener datos recientes de {ticker}: {e}")
            
    if not new_data_frames:
        print("⚠️ No se encontraron nuevos datos para añadir.")
        return
        
    df_new = pd.concat(new_data_frames, ignore_index=True)
    
    # Concatenar con el archivo Parquet histórico si ya existe
    if os.path.exists(parquet_path):
        print(f"📂 Leyendo archivo histórico existente: {parquet_path}")
        df_existing = pd.read_parquet(parquet_path)
        df_existing["date"] = pd.to_datetime(df_existing["date"]).dt.date
        
        # Unir y eliminar duplicados (manteniendo el registro más reciente)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["date", "ticker"], keep="last")
        df_combined = df_combined.sort_values(by=["ticker", "date"]).reset_index(drop=True)
    else:
        print("📁 No se encontró un archivo previo. Creando la base histórica inicial.")
        df_combined = df_new
        
    # Asegurar columnas de partición o metadatos de fecha
    df_combined["year"] = pd.to_datetime(df_combined["date"]).dt.year
    df_combined["month"] = pd.to_datetime(df_combined["date"]).dt.month
    
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    df_combined.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)
    
    print(f"✅ Archivo actualizado con éxito en: {parquet_path}")
    print(f"📊 Total de registros consolidados en el dataset: {len(df_combined):,}")

if __name__ == "__main__":
    watchlist_tickers = [
        "IBE.MC",   # Iberdrola S.A.
        "NEE",      # NextEra Energy Inc.
        "GEV",      # GE Vernova LLC
        "CVX",      # Chevron Corporation
        "XOM",      # ExxonMobil Corporation
        "WTI",      # W&T Offshore Inc.
        "BZ=F"      # Brent Crude Oil
    ]
    update_latest_market_day(watchlist_tickers)
  
