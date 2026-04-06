import pandas as pd

# Load a local CSV file
df = pd.read_csv('data/GLUE/CoLA/train.tsv', sep='\t', on_bad_lines='skip')

print(df.columns.tolist())