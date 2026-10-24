import pandas as pd

# 1. Cargamos los dos archivos CSV
# Asegúrate de que los nombres coincidan con los de tu carpeta
df_historico = pd.read_csv('financial_data1.csv')
df_nuevos_datos = pd.read_csv('financial_data2310.csv')

# 2. Concatenamos ambos DataFrames
df_completo = pd.concat([df_historico, df_nuevos_datos], ignore_index=True)

# 3. Eliminamos duplicados
# Usamos 'datetime' y 'ticker' como referencia. 
# keep='last' asegura que si hay datos en ambos archivos para la misma hora, 
# se quede con el del archivo nuevo (2310).
df_limpio = df_completo.drop_duplicates(subset=['datetime', 'ticker'], keep='last')

# 4. Ordenamos cronológicamente por activo (ticker) y fecha para que quede perfecto
df_limpio = df_limpio.sort_values(by=['ticker', 'datetime']).reset_index(drop=True)

# 5. Guardamos el nuevo archivo CSV listo para usar
df_limpio.to_csv('financial_data_FINAL.csv', index=False)

print(f"¡Proceso terminado! El archivo final tiene {len(df_limpio)} filas.")
print("Ya puedes descargar el archivo 'financial_data_FINAL.csv' de tu carpeta.")
