import pandas as pd


"""
df_sizes = {}
# Load a local CSV file
for x in ["GLUE", "VALUE"]:
     for task in ["CoLA", "MNLI", "QNLI", "QQP", "RTE", "SST-2", "STS-B", "WNLI"]:
            for mode in ["train", "dev"]:
                  if task == "MNLI" and mode == "dev":
                        file_path = f'data/{x}/{task}/{mode}_mismatched.tsv'
                  else: 
                        file_path =f'data/{x}/{task}/{mode}.tsv'
                  df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip')
                  df_sizes[f"{x}_{task}_{mode}"] = len(df)

print(df_sizes)
"""

df = pd.read_csv(f"data/VALUE/QNLI/train.tsv", sep='\t', on_bad_lines='skip')
print(df.columns)