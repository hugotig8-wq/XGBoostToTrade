"""
Módulo de inspección rápida del Data Lake en Parquet usando Polars.
"""

import polars as pl

def inspect_parquet_lake(file_path: str = "data/parquet/financial_data.parquet") -> None:
    """Lee y valida la integridad del archivo Parquet procesado."""
    print("Cargando dataset con Polars...")
    
    # Lectura ultra rápida con Polars
    df = pl.read_parquet(file_path)
    
    print("\n" + "="*50)
    print(f"Esquema del Dataset:")
    print(df.schema)
    print("="*50)
    
    print(f"\nTotal de filas registradas: {df.height}")
    print(f"Total de columnas: {df.width}")
    
    # Resumen de filas por cada ticker
    print("\nDesglose de filas por activo:")
    summary = df.group_by("ticker").agg(
        pl.count("date").alias("total_dias"),
        pl.col("date").min().alias("fecha_inicio"),
        pl.col("date").max().alias("fecha_fin")
    ).sort("ticker")
    
    print(summary)

if __name__ == "__main__":
    inspect_parquet_lake()
  
