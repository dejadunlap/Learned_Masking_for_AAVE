from collections import namedtuple
from transformers import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from transformers.models.distilbert.modeling_distilbert import DistilBertForSequenceClassification
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.distilbert.tokenization_distilbert import DistilBertTokenizer
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
