"""
Módulo de Inspección Específica por Fecha y Hora
------------------------------------------------
Permite consultar el archivo Parquet crudo (/data/parquet/raw/intraday_60d_raw.parquet)
para un timestamp específico (ej. 15 de Octubre a las 12:00 o 14 de Octubre a las 09:00).
"""

import os
import pandas as pd


def inspect_specific_sample(
    target_datetime: str, 
    ticker: str = None,
    parquet_path: str = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/raw/intraday_60d_raw.parquet"
) -> None:
    """
    Busca y muestra los registros del archivo Parquet para una fecha y hora específica,
    opcionalmente filtrando por un ticker determinado.

    Args:
        target_datetime (str): Fecha y hora en formato 'YYYY-MM-DD HH:MM:SS' (ej. '2026-10-15 12:00:00').
        ticker (str, optional): Ticker específico a consultar. Si es None, muestra todos los activos.
        parquet_path (str): Ruta absoluta al archivo Parquet de datos vírgenes.
    """
    if not os.path.exists(parquet_path):
        print(f"⚠️ No se encontró el archivo Parquet en la ruta: {parquet_path}")
        print("💡 Asegúrate de haber ejecutado primero el 'raw_collector.py'.")
        return

    print(f"📂 Leyendo datos desde: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # Asegurar que la columna datetime sea de tipo datetime para comparar de forma precisa
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        
        # Normalizar zona horaria si el DataFrame la incluye para evitar errores de comparación
        if df["datetime"].dt.tz is not None:
            target_dt = pd.to_datetime(target_datetime).tz_localize(df["datetime"].dt.tz)
        else:
            target_dt = pd.to_datetime(target_datetime)
        
        # Filtrar por fecha y hora exacta
        filtered_df = df[df["datetime"] == target_dt]
        
        if filtered_df.empty:
            print(f"⚠️ No hay coincidencia exacta para el timestamp: {target_datetime}. Buscando registros más cercanos...")
            df["time_diff"] = (df["datetime"] - target_dt).abs()
            min_diff = df["time_diff"].min()
            filtered_df = df[df["time_diff"] == min_diff]
            print(f"ℹ️ Mostrando registros más cercanos con una diferencia temporal de {min_diff}:")
    else:
        print("❌ La columna 'datetime' no existe en el esquema del archivo Parquet.")
        return

    if ticker:
        filtered_df = filtered_df[filtered_df["ticker"].str.upper() == ticker.upper()]

    if filtered_df.empty:
        print(f"❌ No se encontraron datos para el criterio especificado.")
        return

    print("\n" + "="*90)
    print(f"🔍 INSPECCIÓN ESPECÍFICA PARA EL TIMESTAMP: {target_datetime}")
    print("="*90)
    print(filtered_df.to_string(index=False))
    print("="*90 + "\n")


if __name__ == "__main__":
    print("--- PRUEBA 1: 15 de Octubre de 2026 a las 12:00 ---")
    inspect_specific_sample(target_datetime="2026-10-15 12:00:00")

    print("\n--- PRUEBA 2: 14 de Octubre de 2026 a las 09:00 ---")
    inspect_specific_sample(target_datetime="2026-10-14 09:00:00")
      
