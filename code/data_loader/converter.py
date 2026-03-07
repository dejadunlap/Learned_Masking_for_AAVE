import pandas as pd
import csv
import uuid
import random

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
tsv_train_file = "./data/train.tsv"
tsv_dev_file = "./data/dev.tsv"
tsv_test_file = "./data/test.tsv"


train_rows = []
dev_rows = []
test_rows = []

for row in df.itertuples(index = True):
    ran = random.random()
    if ran < 0.7: 
        train_rows.append([str(uuid.uuid4()), row.AAVE, 1])
        train_rows.append([str(uuid.uuid4()), row.SAE, 0])
    elif ran >= 0.7 and ran < 0.9: 
        dev_rows.append([str(uuid.uuid4()), row.AAVE, 1])
        dev_rows.append([str(uuid.uuid4()), row.SAE, 0])
    else: 
        test_rows.append([str(uuid.uuid4()), row.AAVE, 1])
        test_rows.append([str(uuid.uuid4()), row.SAE, 0])

with open(tsv_train_file, mode="w", newline="") as f:

    tsv_writer = csv.writer(f, delimiter="\t")
    try: 
        tsv_writer.writerow(header)
        tsv_writer.writerows(train_rows)
    except Exception as e: 
        print(f"Ran into the following error while converting data {e}")

with open(tsv_dev_file, mode="w", newline="") as f:

    tsv_writer = csv.writer(f, delimiter="\t")
    try: 
        tsv_writer.writerow(header)
        tsv_writer.writerows(dev_rows)
    except Exception as e: 
        print(f"Ran into the following error while converting data {e}")

with open(tsv_test_file, mode="w", newline="") as f:

    tsv_writer = csv.writer(f, delimiter="\t")
    try: 
        tsv_writer.writerow(header)
        tsv_writer.writerows(test_rows)
    except Exception as e: 
        print(f"Ran into the following error while converting data {e}")


