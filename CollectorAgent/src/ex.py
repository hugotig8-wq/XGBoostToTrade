import pandas as pd
df = pd.read_parquet("/home/humberto/master_ml_dataset.parquet")
print([c for c in df.columns if "return" in c.lower()])
print(df.shape)
