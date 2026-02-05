import json
import copy
from tqdm import tqdm

# example class formatting ex. from the tsv file directly
class SentenceExample(object):
    def __init__(self, uuid, sent, label=None):
        self.uuid = uuid
        self.sent = sent
        self.label = label
    
    def __repr__(self):
        return str(self.to_json_string())
    
    def to_dict(self):
        return copy.deepcopy(self.__dict__)
    
    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

# bert input class formatting examples into BERT appropriate format
class BertInputFeature(object):
    def __init__(self, uuid, input_ids, attention_mask, token_type_ids, labels=[0,1]):
        self.uuid = uuid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.gold = label

    def __repr__(self):
        return str(self.to_json_string())
    
    def to_dict(self):
        return copy.deepcopy(self.__dict__)
    
    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

# create class to convert dataset into format for processing
class AAVEDataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_examples = self.get_split_examples("train")
        self.val_examples = self.get_split_examples("dev")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.tsv".format(which_split))
        print("INFO {} is looking for {}".format("AAVE", where_))
        return self._create_examples(where_, which_split)
    
    def get_labels(self):
        return ["0", "1"]
    
    def _create_examples(self, input_file, which_split):
        # converts raw sentences into SentenceExample
        sentences_exs = []
        with open(input_file, "r", encoding="utf-8") as f: 
            next(f)
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                uuid, text, label = line[0], line[1], line[2]
                sentences_exs.append(
                    SentenceExample(uuid, text, label)
                )
        return sentences_exs

# function to convert exs from tsv to BERT appropriate format
def example_to_feature(examples, tokenizer, max_seq_len, label_list, pad_token=0, pad_token_segment_id=0):
    features = []

    print("[INFO] Converting our Examples into Features")
    for idx, ex in enumerate(tqdm(examples)):
        #tokenizes the input
        inputs = tokenizer.encode_plus(ex.sent, add_special_tokens=True, max_length=max_seq_len)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        padding_len = max_seq_len - len(input_ids)
        input_ids = input_ids + [pad_token] * padding_len
        attention_mask = attention_mask + [0] * padding_len
        token_type_ids = token_type_ids + [pad_token_segment_id] * padding_len
        assert (
            len(input_ids) == len(attention_mask) == len(token_type_ids) == max_seq_len
        )

        features.append(BertInputFeature(
            uuid = ex.uuid,
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            label=ex.label
        ))

    return features



