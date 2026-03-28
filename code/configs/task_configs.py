from data_loader import SeqClsDataIter, TaggingDataIter, MultipleChoiceDataIter
from transformers.models.bert.tokenization_bert import BertTokenizer


task2main_metric = {
    "aave_mask":    ["accuracy"],
    "boolQ_aave":   ["accuracy"],
    "sst2_aave":    ["accuracy"],
    "multirc_aave": ["f1", "accuracy"],
    "wsc_aave":     ["accuracy"],
    "copa_aave":    ["accuracy"],
    "boolQ_sae":    ["accuracy"],
    "sst2_sae":     ["accuracy"],
    "multirc_sae":  ["f1", "accuracy"],
    "wsc_sae":      ["accuracy"],
    "copa_sae":     ["accuracy"],
}


task2dataiter = {
    "aave_mask": SeqClsDataIter,
    "boolQ_aave": SeqClsDataIter,
    "sst2_aave": SeqClsDataIter,
    "multirc_aave": SeqClsDataIter,
    "copa_aave": SeqClsDataIter,
    "wsc_aave": SeqClsDataIter,
    "boolQ_sae": SeqClsDataIter,
    "sst2_sae": SeqClsDataIter,
    "multirc_sae": SeqClsDataIter,
    "copa_sae": SeqClsDataIter,
    "wsc_sae" : SeqClsDataIter
}