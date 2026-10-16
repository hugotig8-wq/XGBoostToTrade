"""
Módulo de Inspección de Volumen Máximo Diario (inspect_max_volume.py)
---------------------------------------------------------------------
Lee los datos intradiarios en bruto (capa raw) y determina para cada día 
y cada activo la franja horaria exacta en la que se registró el volumen máximo.
"""

import os
import pandas as pd


def inspect_max_volume(
    parquet_path: str = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/raw/financial_data.parquet"
) -> None:
    """
    Carga el archivo Parquet intradiario y encuentra la franja horaria con mayor volumen por día y por ticker.
    """
    if not os.path.exists(parquet_path):
        print(f"⚠️ No se encontró el archivo Parquet en la ruta: {parquet_path}")
        print("💡 Asegúrate de haber ejecutado primero el colector intradiario en la carpeta raw.")
        return

    print(f"📂 Leyendo datos intradiarios desde: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # Verificar columnas necesarias
    required_cols = ["datetime", "ticker", "volume"]
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Falta la columna obligatoria '{col}' en el dataset.")
            return

    # Asegurar tipo datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    # Extraer la fecha (sin hora) para agrupar por día
    df["date"] = df["datetime"].dt.date

    # Encontrar el índice de la fila con el volumen máximo para cada combinación de (ticker, date)
    max_vol_idx = df.groupby(["ticker", "date"])["volume"].idxmax()

    # Filtrar el DataFrame original con esos índices
    df_max_volume = df.loc[max_vol_idx].copy()

    # Ordenar por fecha y ticker para una visualización limpia
    df_max_volume = df_max_volume.sort_values(by=["date", "ticker"], ascending=[False, True]).reset_index(drop=True)

    print("\n" + "=" * 90)
    print("📊 FRANJA HORARIA CON EL VOLUMEN MÁXIMO DE CADA DÍA POR ACTIVO")
    print("=" * 90)
    
    # Mostrar columnas relevantes
    cols_to_show = ["date", "datetime", "ticker", "volume", "close", "high", "low"]
    available_cols = [c for c in cols_to_show if c in df_max_volume.columns]
    
    print(df_max_volume[available_cols].head(28).to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    inspect_max_volume()
  
