import os
import pandas as pd

properties = os.listdir()
for prop in properties:
    if '.py' in prop:
        continue
    csv_files = os.listdir(prop)
    for csv in csv_files:
        df = pd.read_csv(prop+'/'+csv)
        df.columns = ['formula', 'target']
        df['formula'] = df['formula'].str.split("_ICSD").str[0]
        df.dropna(inplace=True)
        df.to_csv(prop+'/'+csv, index=False)
