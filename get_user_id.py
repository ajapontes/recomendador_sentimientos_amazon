import pandas as pd
df = pd.read_parquet('data/processed/electronics_sample_100k.parquet')
print(df['user_id'].iloc[0])
