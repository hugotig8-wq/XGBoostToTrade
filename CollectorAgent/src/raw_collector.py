"""
Módulo CollectorRawAgent
-----------------------
Se encarga exclusivamente de la descarga, captura e inspección de datos vírgenes (Raw)
en velas de 1 hora durante los últimos 60 días para la lista completa de seguimiento (Watchlist).

Ruta de almacenamiento: /workspaces/XGBoostToTrade/CollectorAgent/data/parquet/raw
"""

import os
from datetime import datetime
from typing import List
import pandas as pd
import yfinance as yf


class ColectorRawAgent:
    def __init__(
        self, 
        watchlist: List[str] = None, 
        #output_dir: str = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/raw"
        output_dir: str = "/home/humberto"
    ):
        """
        Inicializa el agente colector virgen con la lista completa de seguimiento de tu Watchlist.
        """
        self.watchlist = watchlist if watchlist else [
            "IBE.MC",   # Iberdrola S.A.
            "NEE",      # NextEra Energy Inc.
            "GEV",      # GE Vernova LLC
            "CVX",      # Chevron Corporation
            "XOM",      # ExxonMobil Corporation
            "WTI",      # W&T Offshore Inc.
            "BZ=F"      # Brent Crude Oil (EuroOIL)
        ]
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_and_save_60d_intraday(self) -> pd.DataFrame:
        """
        Descarga de forma segura velas de 1 hora de los últimos 60 días
        (límite técnico de yfinance) preservando el dato virgen sin indicadores.
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Iniciando ColectorRawAgent para {len(self.watchlist)} activos...")
        
        processed_dfs = []
        failed_tickers = []

        for ticker in self.watchlist:
            try:
                print(f" 📥 Descargando activo virgen: {ticker}...")
                # Límite técnico de yfinance para velas de 1 hora: period="60d", interval="1h"
                df_ticker = yf.download(
                    ticker, 
                    #period="60d",
                    period="8d",
                    interval="1h", 
                    auto_adjust=True, 
                    progress=False
                )
                
                if df_ticker.empty:
                    print(f" ⚠️ Advertencia: No se obtuvieron datos intradiarios para {ticker}")
                    failed_tickers.append(ticker)
                    continue

                # Aplanar MultiIndex si es generado por versiones recientes de yfinance
                if isinstance(df_ticker.columns, pd.MultiIndex):
                    df_ticker.columns = df_ticker.columns.get_level_values(0)

                df_ticker = df_ticker.reset_index()
                df_ticker["ticker"] = ticker
                
                # Normalizar nombres de columnas a minúsculas
                df_ticker.columns = [str(col).lower().replace(" ", "_") for col in df_ticker.columns]
                
                processed_dfs.append(df_ticker)
                
            except Exception as e:
                print(f" ❌ Error al descargar {ticker}: {e}")
                failed_tickers.append(ticker)

        if not processed_dfs:
            print("❌ No se pudo procesar ningún activo de la lista en este ciclo.")
            return pd.DataFrame()

        # Consolidar todos los DataFrames
        final_df = pd.concat(processed_dfs, ignore_index=True)

        # Ruta final del archivo virgen Parquet
        parquet_path = os.path.join(self.output_dir, "financial_data2310.parquet")

        # Guardado en formato Parquet comprimido con Snappy
        #final_df.to_parquet(
            #parquet_path, engine="pyarrow", compression="snappy", index=False
        #)

        final_df.to_csv(
            parquet_path
        )
        #print(final_df.to_string())

        print("\n" + "="*60)
        print("✅ RESGUARDO DE DATOS VÍRGENES (60 días - 1h) COMPLETADO")
        print("="*60)
        print(f" Timestamp de ejecución      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Activos procesados          : {len(self.watchlist) - len(failed_tickers)} / {len(self.watchlist)}")
        print(f" Filas totales almacenadas   : {len(final_df):,}")
        print(f" Archivo Parquet guardado en : {parquet_path}")
        if failed_tickers:
            print(f" Activos fallidos            : {failed_tickers}")
        print("="*60 + "\n")

        return final_df

    

if __name__ == "__main__":
    # Instanciar y ejecutar el agente colector
    agent = ColectorRawAgent()
    
    # 1. Descargar y resguardar en la ruta especificada
    raw_dataframe = agent.fetch_and_save_60d_intraday()
    
