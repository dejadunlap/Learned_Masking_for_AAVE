from data_loader import SeqClsDataIter

task2main_metric = {
    "aave_mask": ["mcc"],
    "sae_mask": ["mcc"],
    "sae_aave_mask": ["mcc"],
    "sst2_aave": ["accuracy"],
    "mnli_aave": ["accuracy"],
    "qqp_aave": ["accuracy"],
    "cola_aave": ["mcc"],
    "qnli_aave": ["accuracy"],
    "rte_aave": ["accuracy"],
    "wnli_aave": ["accuracy"],
    "stsb_aave" : ["pearson"],
    "sst2_sae": ["accuracy"],
    "mnli_sae": ["accuracy"],
    "qqp_sae": ["accuracy"],
    "cola_sae": ["mcc"],
    "qnli_sae": ["accuracy"],
    "rte_sae": ["accuracy"],
    "wnli_sae": ["accuracy"],
    "stsb_sae" : ["pearson"],
}


task2dataiter = {
    "aave_mask": SeqClsDataIter,
    "sst2_aave": SeqClsDataIter,
    "mnli_aave": SeqClsDataIter,
    "qqp_aave": SeqClsDataIter,
    "cola_aave": SeqClsDataIter,
    "qnli_aave": SeqClsDataIter,
    "rte_aave": SeqClsDataIter,
    "wnli_aave": SeqClsDataIter,
    "stsb_aave" : SeqClsDataIter,
    "sst2_sae": SeqClsDataIter,
    "mnli_sae": SeqClsDataIter,
    "qqp_sae": SeqClsDataIter,
    "cola_sae": SeqClsDataIter,
    "qnli_sae": SeqClsDataIter,
    "rte_sae": SeqClsDataIter,
    "wnli_sae": SeqClsDataIter,
    "stsb_sae" : SeqClsDataIter,
}