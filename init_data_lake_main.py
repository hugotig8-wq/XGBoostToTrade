import os
from datetime import datetime
import pandas as pd
import yfinance as yf


def fetch_and_save_to_parquet(
    tickers: list[str], period: str = "5y", output_dir: str = "data/parquet"
):
    """Descarga datos de yfinance y los guarda en formato Parquet."""
    print(f"Descargando datos para los tickers: {tickers} (período: {period})...")

    # 1. Extracción con yfinance
    df = yf.download(
        tickers, period=period, interval="1d", group_by="ticker", auto_adjust=True
    )

    if df.empty:
        print("No se encontraron datos.")
        return

    # 2. Transformación y Aplanado (Manejo de MultiIndex si son varios tickers)
    processed_dfs = []

    if len(tickers) == 1:
        ticker = tickers[0]
        df_single = df.reset_index()
        df_single["Ticker"] = ticker
        processed_dfs.append(df_single)
    else:
        for ticker in tickers:
            if ticker in df.columns.levels[0]:
                df_ticker = df[ticker].dropna(how="all").reset_index()
                df_ticker["Ticker"] = ticker
                processed_dfs.append(df_ticker)

    final_df = pd.concat(processed_dfs, ignore_index=True)

    # 3. Limpieza y estandarización de nombres de columnas
    final_df.columns = [str(col).lower().replace(" ", "_") for col in final_df.columns]

    # Convertir fecha a datetime explícito y añadir columnas de particionamiento
    final_df["date"] = pd.to_datetime(final_df["date"])
    final_df["year"] = final_df["date"].dt.year
    final_df["month"] = final_df["date"].dt.month

    # 4. Asegurar directorio de salida
    os.makedirs(output_dir, exist_ok=True)

    # 5. Guardar en Parquet (Particionado por Año/Mes o Archivo Único)
    parquet_path = os.path.join(output_dir, "financial_data.parquet")

    final_df.to_parquet(
        parquet_path, engine="pyarrow", compression="snappy", index=False
    )

    print(f"✅ Ingesta completada con éxito.")
    print(f"Total registros ingestados: {len(final_df)}")
    print(f"Archivo guardado en: {parquet_path}\n")

    # Verificación de lectura rápida
    df_read = pd.read_parquet(parquet_path)
    print("Muestra de datos guardados:")
    print(df_read.head())


if __name__ == "__main__":
    # Tickers de prueba
    test_tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", TSLA, SHA0, 6324, 6268, 6594]
    fetch_and_save_to_parquet(tickers=test_tickers, period="5y")
          
