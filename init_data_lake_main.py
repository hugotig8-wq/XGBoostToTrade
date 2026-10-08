import os
import pandas as pd
import yfinance as yf


def fetch_and_save_to_parquet(
    tickers: list[str], period: str = "5y", output_dir: str = "data/parquet"
) -> None:
    """Descarga datos de yfinance de forma segura y los almacena en Parquet."""
    print(f"Iniciando descarga para {len(tickers)} tickers (período: {period})...")
    
    processed_dfs = []
    failed_tickers = []

    # Iterar individualmente para manejo limpio de excepciones por ticker
    for ticker in tickers:
        try:
            print(f"Descargando {ticker}...")
            df_ticker = yf.download(
                ticker, period=period, interval="1d", auto_adjust=True, progress=False
            )
            
            if df_ticker.empty:
                print(f"⚠️ Advertencia: No se encontraron datos para {ticker}")
                failed_tickers.append(ticker)
                continue

            # Aplanar MultiIndex si se genera automáticamente
            if isinstance(df_ticker.columns, pd.MultiIndex):
                df_ticker.columns = df_ticker.columns.get_level_values(0)

            df_ticker = df_ticker.reset_index()
            df_ticker["Ticker"] = ticker
            
            # Normalizar nombres de columnas a minúsculas
            df_ticker.columns = [str(col).lower().replace(" ", "_") for col in df_ticker.columns]
            
            processed_dfs.append(df_ticker)
            
        except Exception as e:
            print(f"❌ Error al descargar {ticker}: {e}")
            failed_tickers.append(ticker)

    if not processed_dfs:
        print("No se pudo procesar ningún ticker.")
        return

    # Consolidar todos los DataFrames
    final_df = pd.concat(processed_dfs, ignore_index=True)

    # Añadir campos de particionamiento
    final_df["date"] = pd.to_datetime(final_df["date"])
    final_df["year"] = final_df["date"].dt.year
    final_df["month"] = final_df["date"].dt.month

    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    parquet_path = os.path.join(output_dir, "financial_data.parquet")

    # Guardar en formato Parquet
    final_df.to_parquet(
        parquet_path, engine="pyarrow", compression="snappy", index=False
    )

    print("\n" + "="*50)
    print("✅ Ingesta a Parquet completada con éxito.")
    print(f"Filas totales almacenadas : {len(final_df)}")
    print(f"Archivo guardado en      : {parquet_path}")
    if failed_tickers:
        print(f"Tickers con error        : {failed_tickers}")
    print("="*50)


if __name__ == "__main__":
    # Tickers con prefijos/sufijos de bolsa corregidos
    test_tickers = [
        "AAPL", "MSFT", "GOOGL", "NVDA", "TSLA",
        "SHA0.DE", "6324.T", "6268.T", "6594.T"
    ]
    fetch_and_save_to_parquet(tickers=test_tickers, period="5y")
        
