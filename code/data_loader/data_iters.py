from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from .bert_formatting import *
import torch
import os
import pickle
import json
import uuid

# consider mcc for imbalanced datasets
class SeqClsDataIter(object):
    def __init__(self, model, tokenizer, max_seq_len, dataset = AAVEDataset("data/")):
        self.pdata = AAVEDataset("data/")
        self.max_seq_len = max_seq_len
        self.metrics = ["accuracy"]
        self.num_labels = 2
        self.model = model
        self.train_examples = self.wrap_iter(model, "train", self.pdata.train_examples, tokenizer, max_seq_len)
        self.val_examples = self.wrap_iter(model, "dev", self.pdata.val_examples, tokenizer, max_seq_len)
        self.test_examples = self.wrap_iter(model, "test", self.pdata.test_examples, tokenizer, max_seq_len)

    def wrap_iter(self, model, split, examples, tokenizer, max_seq_len):
        cached_ = os.path.join(
            "data", "cached", f"cached_{self.model}_{self.max_seq_len}_{uuid.uuid4()}.pkl"
        )
        meta_ = cached_.replace(".pkl", ".meta")
        fts = example_to_feature(examples, tokenizer, max_seq_len, self.label_list)
        uid, complete = str(uuid.uuid4()), True
        try:
            with open(cached_, "wb") as f:
                pickle.dump({"fts":fts, "uid":uid}, f)
        except: 
            complete = False
        with open(meta_, "w") as f:
            json.dump({"complete":complete, "uid":uid}, f)
        return _SeqClstIter(fts)

    @property
    def label_list(self):
        return self.pdata.get_labels()
    


class _SeqClstIter(torch.utils.data.Dataset):
    def __init__(self, fts):
        print(vars(fts[0]))
        self.uides = [ft.uuid for ft in fts]
        self.golds = torch.as_tensor([int(ft.gold) for ft in fts], dtype=torch.long)       
        self.input_ids = torch.as_tensor([ft.input_ids for ft in fts], dtype=torch.long)
        self.attention_mask = torch.as_tensor([ft.attention_mask for ft in fts], dtype=torch.long)
        self.token_type_ids = torch.as_tensor([ft.token_type_ids for ft in fts], dtype=torch.long)
    
    def __getitem__(self, idx):
        return (self.uides[idx], self.input_ids[idx], self.golds[idx], self.attention_mask[idx], self.token_type_ids[idx])
    
    def __len__(self):
        return len(self.uides)

