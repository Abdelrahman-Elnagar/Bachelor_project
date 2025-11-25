import pandas as pd

df = pd.read_csv(r'Data\Germany_aggregation\wind_germany\Germany_total.csv', 
                 usecols=['datetime', 'wind_speed'], 
                 parse_dates=['datetime'])

df_2024 = df[df['datetime'].dt.year == 2024]

print(f'Total records: {len(df_2024)}')
print(f'Unique datetimes: {df_2024["datetime"].nunique()}')
print(f'\nFirst 10 records:')
print(df_2024.head(10))
print(f'\nRecords per datetime (sample):')
print(df_2024.groupby('datetime').size().head(10))

