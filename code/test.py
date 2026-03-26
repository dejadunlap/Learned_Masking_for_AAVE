from data_loader import SeqClsDataIter
import configs.task_configs as task_configs
from data_loader.data_iters import *
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.bert.tokenization_bert import BertTokenizer
from trainers.finetuner import BertFinetuner
from trainers.hooks import EvaluationRecorder, SparsityRecorder
import utils.logging as logging



this_task = "boolQ_aave"

conf = dict(
    ptl="bert",
    model="bert-base-uncased",
    task=this_task,
    model_scheme="vector_cls_sentence",
    experiment="debug",
    max_seq_len=128,
    lr=1e-3,
    world="0",
    batch_size=32,
    eval_every_batch=60,
    num_epochs=10,
    do_BL=False,
    do_MS=True,
    ptl_req_grad=False,
    classifier_req_grad=False,
    mask_classifier=True,
    mask_ptl=True,
    layers_to_mask="2,3,4,5,6,7,8,9,10,11",
    train_fast=True,
    num_snapshots=10,
    masking_scheduler_conf="lambdas_lr=0,sparsity_warmup=automated_gradual_sparsity,final_sparsity=0.05,sparsity_warmup_interval_epoch=0.1,init_epoch=0,final_epoch=1,initial_sparsity=0.8",
    checkpoint_root="tmp/jobs/"
)

pdata = task2dataset[conf["task"]](task2datadir[conf["task"]])

tokenizer = BertTokenizer.from_pretrained(conf["model"])
data_iter = task_configs.task2dataiter[conf["task"]](conf["task"], conf["model"], tokenizer, conf["max_seq_len"])
#task_data_iter = data_iter(conf["task"], conf["model"], tokenizer, conf["max_seq_len"])

#trainer = BertFinetuner(conf, logger=None, data_iter=task_data_iter)

train_a = [text.text_a for text in data_iter.pdata.trn_egs]
val_a = set([text.text_b for text in data_iter.pdata.val_egs])

#
# same_train_val = train_a.intersection(val_a)
print(train_a[:5])
        
