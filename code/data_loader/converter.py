import pandas as pd
import csv
import uuid

"""
File convert the original dataset compiled from other sources and convert to expected format with AAVE/SAE labels for mask training
"""
file_path = "./data/orig_data.tsv"

try:
    df = pd.read_csv(file_path, sep="\t", header=0)

    print(df.head())
except FileNotFoundError: 
    print(f"File not found at this path {file_path}")
except Exception as e: 
    print(f"Got the following error: {e}")

header = ["id", "sentence", "label"]
tsv_file = "./data/converted_data.tsv"
rows = []

for row in df.itertuples(index = True):
    rows.append([str(uuid.uuid4()), row.AAVE, 1])
    rows.append([str(uuid.uuid4()), row.SAE, 0])

print(rows[:5])

with open(tsv_file, mode="w", newline="") as f:

    tsv_writer = csv.writer(f, delimiter="\t")
    try: 
        tsv_writer.writerow(header)
        tsv_writer.writerows(rows)
    except Exception as e: 
        print(f"Ran into the following error while converting data {e}")
