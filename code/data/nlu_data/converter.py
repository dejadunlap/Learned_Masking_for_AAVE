import pandas as pd
import csv
import uuid
import random
import numpy as np

for task in ["boolQ", "sst2", "multirc", "wsc", "copa"]:
    file_path = f"data/nlu_data/{task}/orig_{task}.csv"
    df = pd.read_csv(file_path, sep=",", header=0)

    aave_rows = []
    sae_rows = []

    for idx, row in df.iterrows():
        if task == "multirc":
            aave_rows.append([str(uuid.uuid4()), row['Translated Paragraph'], row['Translated Question'] + " " + str(row['Translated Answer']), row['Actual Label']])
            sae_rows.append([str(uuid.uuid4()), row['Paragraph'], row['Question'] + " " + str(row['Answer']), row['Actual Label']])

        elif task == "sst2":
            label = "0" if row['Original Sentiment'] == "Negative" else "1"
            aave_rows.append([str(uuid.uuid4()), row['Translated'], label])
            sae_rows.append([str(uuid.uuid4()), row['Original'], label])

        elif task == "boolQ":
            label = "1" if row['actual answer'] == "TRUE" else "0"
            aave_rows.append([str(uuid.uuid4()), row['aave passage'], row['aave question'], label])
            sae_rows.append([str(uuid.uuid4()), row['se passage'], row['se question'], label])

        elif task == "copa":
            # each premise gets two rows: one per choice
            # label=0 means this choice is wrong, label=1 means correct
            correct = row['Actual Label']  # 0 or 1, indicating which choice is correct
            aave_rows.append([str(uuid.uuid4()), row['Translated Premise'], row['Translated Choice1'], "1" if correct == "0" else '0'])
            aave_rows.append([str(uuid.uuid4()), row['Translated Premise'], row['Translated Choice2'], "1" if correct == '1' else '0'])
            sae_rows.append([str(uuid.uuid4()), row['Premise'], row['Choice1'], '1' if correct == '0' else '0'])
            sae_rows.append([str(uuid.uuid4()), row['Premise'], row['Choice2'], '1' if correct == '1' else '0'])

        elif task == "wsc":
            # each passage gets two rows: one per span candidate
            correct = row['Actual Reference']  # 0 or 1, indicating which span the pronoun refers to
            sae_rows.append([str(uuid.uuid4()), row['Original Passage'], row['Span1'], '1' if correct == '0' else '0'])
            sae_rows.append([str(uuid.uuid4()), row['Original Passage'], row['Span2'], '1' if correct == '1' else '0'])
            aave_rows.append([str(uuid.uuid4()), row['Translated Passage'], row['Translated Span1'], '1' if correct == '0' else '0'])
            aave_rows.append([str(uuid.uuid4()), row['Translated Passage'], row['Translated Span2'], '1' if correct == '1' else '0'])

    random.shuffle(sae_rows)
    random.shuffle(aave_rows)

    def split(rows):
        n = len(rows)
        return (
            rows[:int(n * 0.7)],
            rows[int(n * 0.7):int(n * 0.9)],
            rows[int(n * 0.9):]
        )

    train_sae, dev_sae, test_sae = split(sae_rows)
    train_aave, dev_aave, test_aave = split(aave_rows)

    splits = {
        "train": {"sae": train_sae, "aave": train_aave},
        "dev":   {"sae": dev_sae,   "aave": dev_aave},
        "test":  {"sae": test_sae,  "aave": test_aave},
    }

    # header depends on whether task is sentence-pair or single-sentence
    if task in ["multirc", "boolQ", "copa", "wsc"]:
        header = ["id", "text_a", "text_b", "label"]
    else:
        header = ["id", "sentence", "label"]

    for dialect in ["sae", "aave"]:
        for mode in ["train", "dev", "test"]:
            file_path = f"data/nlu_data/{task}/{dialect}/{mode}.tsv"

            with open(file_path, mode="w", newline="", encoding="utf-8") as f:
                tsv_writer = csv.writer(f, delimiter="\t")
                tsv_writer.writerow(header)
                try:
                    tsv_writer.writerows(splits[mode][dialect])
                except Exception as e:
                    print(f"Error writing {task}/{dialect}/{mode}: {e}")