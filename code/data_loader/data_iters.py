from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from .bert_formatting import (
    glue_example_to_feature,
    tagging_example_to_feature,
    multiplechoice_example_to_feature,
)
from .datasets import *
import torch
import os
import pickle
import json
import uuid

task2datadir = {
    "aave_mask": "data/aave_data",
    "cola_aave": "data/VALUE/CoLA",
    "mnli_aave": "data/VALUE/MNLI",
    "qnli_aave": "data/VALUE/QNLI",
    "qqp_aave": "data/VALUE/QQP",
    "rte_aave": "data/VALUE/RTE",
    "sst2_aave": "data/VALUE/SST-2",
    "stsb_aave": "data/VALUE/STS-B",
    "wnli_aave": "data/VALUE/WNLI",
    "cola_sae": "data/GLUE/CoLA",
    "mnli_sae": "data/GLUE/MNLI",
    "qnli_sae": "data/GLUE/QNLI",
    "qqp_sae": "data/GLUE/QQP",
    "rte_sae": "data/GLUE/RTE",
    "sst2_sae": "data/GLUE/SST-2",
    "stsb_sae": "data/GLUE/STS-B",
    "wnli_sae": "data/GLUE/WNLI",
}

task2dataset = {
    "aave_mask": AAVEDataset,
    "sst2_aave": SST2Dataset,
    "cola_aave": COLADataset,
    "mnli_aave": MNLIDataset,
    "qnli_aave": QNLIDataset,
    "qqp_aave": QQPDataset,
    "rte_aave": RTEDataset,
    "stsb_aave": STSBDataset,
    "wnli_aave": WNLIDataset,
    "stsb_aave": STSBDataset,
    "sst2_sae": SST2Dataset,
    "cola_sae": COLADataset,
    "mnli_sae": MNLIDataset,
    "qnli_sae": QNLIDataset,
    "qqp_sae": QQPDataset,
    "rte_sae": RTEDataset,
    "stsb_sae": STSBDataset,
    "wnli_sae": WNLIDataset,
    "stsb_sae": STSBDataset,
}


task2metrics = {
    "aave_mask" : ["accuracy"],
    "sst2_aave": ["accuracy"],
    "mnli_aave": ["accuracy"],
    "qqp_aave": ["f1", "accuracy"],
    "cola_aave": ["mcc"],
    "qnli_aave": ["accuracy"],
    "rte_aave": ["accuracy"],
    "wnli_aave": ["accuracy"],
    "stsb_aave" : ["pearson"],
    "sst2_sae": ["accuracy"],
    "mnli_sae": ["accuracy"],
    "qqp_sae": ["f1", "accuracy"],
    "cola_sae": ["mcc"],
    "qnli_sae": ["accuracy"],
    "rte_sae": ["accuracy"],
    "wnli_sae": ["accuracy"],
    "stsb_sae" : ["pearson"],
}


class SeqClsDataIter(object):
    def __init__(self, task, model, tokenizer, max_seq_len):
        self.task = task
        self.metrics = task2metrics[task]
        self.pdata = task2dataset[task](task2datadir[task])

        # extending class to accomodate stsb task
        if "stsb" in task: 
            self.num_labels = 1
        else:
            self.num_labels = len(self.pdata.get_labels())

        print(f"Train examples {len(self.pdata.trn_egs)}")
        self.trn_dl = self.wrap_iter(
            task, model, "trn", self.pdata.trn_egs, tokenizer, max_seq_len
        )
        
        self.val_dl = self.wrap_iter(
            task, model, "val", self.pdata.val_egs, tokenizer, max_seq_len
        )
        if hasattr(self.pdata, "tst_egs"):
            self.tst_dl = self.wrap_iter(
                task, model, "tst", self.pdata.tst_egs, tokenizer, max_seq_len
            )

    def wrap_iter(self, task, model, split, egs, tokenizer, max_seq_len):
        
        cached_ = os.path.join(
            "data", "cached", f"{task},{max_seq_len},{model},{split},cached.pkl"
        )
        meta_ = cached_.replace(".pkl", ".meta")
        print("[INFO] computing fresh dataset.")
        fts = glue_example_to_feature(
                self.task, egs, tokenizer, max_seq_len, self.label_list
        )
        uid, complete = str(uuid.uuid4()), True
        try:
            with open(cached_, "wb") as f:
                pickle.dump({"fts": fts, "uid": uid}, f)
        except:
            complete = False
        with open(meta_, "w+") as f:
            json.dump({"complete": complete, "uid": uid}, f)
        return _SeqClsIter(fts)

    @property
    def name(self):
        return self.pdata.name

    @property
    def label_list(self):
        return self.pdata.get_labels()


class _SeqClsIter(torch.utils.data.Dataset):
    def __init__(self, fts):
        self.uides = [ft.uid for ft in fts]
        self.input_idses = torch.as_tensor(
            [ft.input_ids for ft in fts], dtype=torch.long
        )
        self.golds = torch.as_tensor([ft.gold for ft in fts], dtype=torch.long)
        self.attention_maskes = torch.as_tensor(
            [ft.attention_mask for ft in fts], dtype=torch.long
        )
        self.token_type_idses = torch.as_tensor(
            [ft.token_type_ids for ft in fts], dtype=torch.long
        )

    def __len__(self):
        return self.golds.shape[0]

    def __getitem__(self, idx):
        return (
            self.uides[idx],
            self.input_idses[idx],
            self.golds[idx],
            self.attention_maskes[idx],
            self.token_type_idses[idx],
        )