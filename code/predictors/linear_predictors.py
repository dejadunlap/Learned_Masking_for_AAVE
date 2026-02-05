from collections import namedtuple
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_bert import BertForSequenceClassification
from transformers.modeling_roberta import RobertaForSequenceClassification
from transformers.modeling_distilbert import DistilBertForSequenceClassification
from transformers.configuration_roberta import RobertaConfig
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_roberta import RobertaTokenizer
from transformers.tokenization_distilbert import DistilBertTokenizer
import torch 
import torch.nn as nn


# setting up classification task for AAVE/SAE identification
class LinearPredictor(BertPreTrainedModel):
    def __init__(self, bert_config, out_dim, dropout):
        super(LinearPredictor, self).__init__(bert_config)
        self.bert = BertModel(bert_config)
        self.classifier = nn.Linear(768, out_dim)
        self.dropout = dropout
        self.init_weights()

    def forward(self):
        raise NotImplementedError
