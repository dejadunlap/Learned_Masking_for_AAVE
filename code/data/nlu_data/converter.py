import pandas as pd
import csv
import uuid
import random
import numpy as np

for task in ["boolQ", "sst2", "multirc", "wsc", "copa"]:
    file_path = f"data/nlu_data/{task}/orig_{task}.csv"
    df = pd.read_csv(file_path, sep=",", header=0)
    splits = {}

    if task == "multirc":
            # group by paragraph first, then assign whole passages to splits
            paragraphs = df['Paragraph'].unique()
            np.random.shuffle(paragraphs)
            
            n = len(paragraphs)
            train_passages = set(paragraphs[:int(n * 0.7)])
            dev_passages   = set(paragraphs[int(n * 0.7):int(n * 0.9)])
            test_passages  = set(paragraphs[int(n * 0.9):])
            
            # 16 passages splits as: ~11 train, ~3 dev, ~2 test
            train_sae_rows, dev_sae_rows, test_sae_rows = [], [], []
            train_aave_rows, dev_aave_rows, test_aave_rows = [], [], []
            
            for _, row in df.iterrows():
                sae_row  = [str(uuid.uuid4()), row['Paragraph'] + " " +row['Question'], str(row['Answer']), row['Actual Label']]
                aave_row = [str(uuid.uuid4()), row['Translated Paragraph'] + " " +row['Translated Question'], str(row['Translated Answer']), row['Actual Label']]
                
                if row['Paragraph'] in train_passages:
                    train_sae_rows.append(sae_row)
                    train_aave_rows.append(aave_row)
                elif row['Paragraph'] in dev_passages:
                    dev_sae_rows.append(sae_row)
                    dev_aave_rows.append(aave_row)
                else:
                    test_sae_rows.append(sae_row)
                    test_aave_rows.append(aave_row)

            # skip the generic shuffle/split below for this task
            splits |= {
                "train": {"sae": train_sae_rows, "aave": train_aave_rows},
                "dev":   {"sae": dev_sae_rows,   "aave": dev_aave_rows},
                "test":  {"sae": test_sae_rows,  "aave": test_aave_rows},
            }
            
            #write directly to train/test/val files
            header = ["id", "text_a", "text_b", "label"]

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

    elif task == "wsc":
            passages = df['Original Passage'].unique()
            np.random.shuffle(passages)
            n = len(passages)
            train_passages = set(passages[:int(n * 0.7)])
            dev_passages   = set(passages[int(n * 0.7):int(n * 0.9)])
            test_passages  = set(passages[int(n * 0.9):])

            train_sae_rows, dev_sae_rows, test_sae_rows = [], [], []
            train_aave_rows, dev_aave_rows, test_aave_rows = [], [], []

            for _, row in df.iterrows():
                correct = row['Actual Reference']  # 0=span1 correct, 1=span2 correct

                sae_span1  = [str(uuid.uuid4()), row['Original Passage'],
                            row['Span1'], 1 if correct == 0 else 0]
                sae_span2  = [str(uuid.uuid4()), row['Original Passage'],
                            row['Span2'], 1 if correct == 1 else 0]
                aave_span1 = [str(uuid.uuid4()), row['Translated Passage'],
                            row['Translated Span1'], 1 if correct == 0 else 0]
                aave_span2 = [str(uuid.uuid4()), row['Translated Passage'],
                            row['Translated Span2'], 1 if correct == 1 else 0]

                if row['Original Passage'] in train_passages:
                    train_sae_rows.extend([sae_span1, sae_span2])
                    train_aave_rows.extend([aave_span1, aave_span2])
                elif row['Original Passage'] in dev_passages:
                    dev_sae_rows.extend([sae_span1, sae_span2])
                    dev_aave_rows.extend([aave_span1, aave_span2])
                else:
                    test_sae_rows.extend([sae_span1, sae_span2])
                    test_aave_rows.extend([aave_span1, aave_span2])

            # skip the generic shuffle/split below for this task
            splits |= {
                "train": {"sae": train_sae_rows, "aave": train_aave_rows},
                "dev":   {"sae": dev_sae_rows,   "aave": dev_aave_rows},
                "test":  {"sae": test_sae_rows,  "aave": test_aave_rows},
            }
            
            #write directly to train/test/val files
            header = ["id", "text_a", "text_b", "label"]

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
    
    else: 
        aave_rows = []
        sae_rows = []

        for idx, row in df.iterrows():
            if task == "sst2":
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

        splits |= {
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