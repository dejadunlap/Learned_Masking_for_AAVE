from data_loader import SeqClsDataIter, TaggingDataIter, MultipleChoiceDataIter
from transformers.models.bert.tokenization_bert import BertTokenizer


task2main_metric = {
    "aave_mask":    ["accuracy"],
    "boolQ_aave":   ["accuracy"],
    "boolQ_random_aave":   ["accuracy"],
    "sst2_aave":    ["accuracy"],
    "multirc_aave": ["f1", "accuracy"],
    "wsc_aave":     ["accuracy"],
    "copa_aave":    ["accuracy"],
    "copa_random_aave":    ["accuracy"],
    "boolQ_sae":    ["accuracy"],
    "boolQ_random_sae":    ["accuracy"],
    "sst2_sae":     ["accuracy"],
    "multirc_sae":  ["f1", "accuracy"],
    "wsc_sae":      ["accuracy"],
    "copa_sae":     ["accuracy"],
    "copa_random_sae":     ["accuracy"],
}


task2dataiter = {
    "aave_mask": SeqClsDataIter,
    "boolQ_aave": SeqClsDataIter,
    "boolQ_random_aave": SeqClsDataIter,
    "sst2_aave": SeqClsDataIter,
    "multirc_aave": SeqClsDataIter,
    "copa_aave": SeqClsDataIter,
    "copa_random_aave": SeqClsDataIter,
    "wsc_aave": SeqClsDataIter,
    "boolQ_sae": SeqClsDataIter,
    "boolQ_random_sae": SeqClsDataIter,
    "sst2_sae": SeqClsDataIter,
    "multirc_sae": SeqClsDataIter,
    "copa_sae": SeqClsDataIter,
    "copa_random_sae": SeqClsDataIter,
    "wsc_sae" : SeqClsDataIter
}