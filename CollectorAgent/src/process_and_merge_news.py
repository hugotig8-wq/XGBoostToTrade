"""
Módulo de Procesamiento y Consolidación de Noticias (process_and_merge_news.py)
-----------------------------------------------------------------------------
Lee los 3 lotes CSV de noticias, los concatena en un único DataFrame,
separa los activos múltiples en filas individuales (explode) y los exporta 
en formato Parquet optimizado para su posterior merge con la capa raw.
"""

import os
import pandas as pd


def process_and_merge_news_batches(
    input_files: list[str] = [
        "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/principal_news_1.csv",
        "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/principal_news_2.csv",
        "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/principal_news_3.csv"
    ],
    output_parquet: str = "/workspaces/XGBoostToTrade/CollectorAgent/data/parquet/cleaned_sentiment_features.parquet"
) -> None:
    """Concatena los 3 lotes de noticias, realiza el explode de activos y guarda el Parquet maestro."""
    
    dfs = []
    for file_path in input_files:
        if os.path.exists(file_path):
            print(f"📂 Leyendo archivo: {file_path}")
            df_temp = pd.read_csv(file_path)
            dfs.append(df_temp)
        else:
            print(f"⚠️ Advertencia: No se encontró el archivo {file_path}. Se omitirá.")

    if not dfs:
        print("❌ No se encontró ningún archivo CSV de noticias para procesar.")
        return

    # 1. Concatenar los 3 DataFrames en uno solo
    print("🔗 Concatenando los 3 lotes en un único DataFrame...")
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"📊 Registros totales combinados antes de limpiar: {len(df_combined):,}")

    # 2. Normalizar la columna affected_assets para dejar exactamente un activo por fila
    if "affected_assets" in df_combined.columns:
        print("⚙️ Normalizando 'affected_assets' (separando por comas y aplicando explode)...")
        # Asegurar tipo string y separar por comas
        df_combined["affected_assets"] = df_combined["affected_assets"].astype(str).str.split(",")
        
        # Duplicar filas por cada activo contenido en la lista
        df_combined = df_combined.explode("affected_assets")
        
        # Limpiar espacios en blanco sobrantes
        df_combined["affected_assets"] = df_combined["affected_assets"].str.strip()
    else:
        print("❌ Error crítico: La columna 'affected_assets' no existe en los datos.")
        return

    # 3. Asegurar directorio de salida y guardar como Parquet optimizado
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df_combined.to_parquet(output_parquet, engine="pyarrow", compression="snappy", index=False)

    print("\n" + "="*60)
    print("✅ PROCESAMIENTO Y FUSIÓN DE NOTICIAS COMPLETADO")
    print("="*60)
    print(f" Filas finales normalizadas (1 por activo): {len(df_combined):,}")
    print(f" Archivo guardado en                      : {output_parquet}")
    print("="*60 + "\n")


if __name__ == "__main__":
    process_and_merge_news_batches()

