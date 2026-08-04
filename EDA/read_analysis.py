import pandas as pd

path = "/Users/krisha/Projs/Neuro-brain-states/env/EDA/EEG_all_patients.csv"

df = pd.read_csv(path)
print(df.head(10))