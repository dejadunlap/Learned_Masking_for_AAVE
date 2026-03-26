import os
from .common import SentencePairExample, SentenceExample

__all__ = [
    "AAVEDataset",
    "BoolQDataset",
    "SST2Dataset",
    "MultiRCDataset",
    "WSCDataset",
    "COPADataset"
]

class GlueDataset(object):
    def get_split_examples(self, split):
        raise NotImplementedError

    def get_labels(self):
        raise NotImplementedError

    def _create_examples(self):
        raise NotImplementedError

# class for initial AAVE/SAE classification task to learn computational represetation of AAVE
class AAVEDataset(object):
    def __init__(self, data_dir):
        self.name = "aave_mask"
        self.data_dir = data_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev")
        self.tst_egs = self.get_split_examples("test")

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
                try:
                    uuid, text, label = line[0], line[1], line[2]
                    sentences_exs.append(
                        SentenceExample(uuid, text, label)
                    )
                except: 
                    continue
        return sentences_exs

class SST2Dataset(object):
    def __init__(self, data_dir):
        self.name = "sst2"
        self.data_dir = data_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev")
        self.tst_egs = self.get_split_examples("test")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.tsv".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, input_file, which_split):
        """ parse and convert raw string to SentenceExample """
        sentence_egs = []
        with open(input_file, "r", encoding="utf-8") as f:
            next(f)
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                if len(line) < 3: 
                    continue
                uid, text, label = line[0], line[1], line[2]
                sentence_egs.append(
                    SentenceExample(uuid=uid, sent=text, label=label)
                )
        return sentence_egs

class BoolQDataset(object):
    def __init__(self, data_dir):
        self.name = "boolQ"
        self.data_dir = data_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev")
        self.tst_egs = self.get_split_examples("test")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.tsv".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, input_file, which_split):
        """ parse and convert raw string to SentencePairExample """
        examples = []
        with open(input_file, "r", encoding="utf-8") as f:
            next(f)
            for idx, line in enumerate(f):
                segs = line.strip().split("\t")
                uid, question, passage, label = segs[0], segs[1], segs[2], segs[3]
                examples.append(
                    SentencePairExample(uuid=uid, text_a=passage, text_b=question, label=label)
                )
        return examples


class BoolQDataset(object):
    """
    BoolQ: boolean QA. Each example is a (passage, question) pair
    with a binary true/false label.
    TSV format: index \t question \t passage \t label
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        print(f"My data dir is {data_dir}")
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev")
        self.tst_egs = self.get_split_examples("test")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, f"{which_split}.tsv")
        print(f"I'm loading data from {where_}")
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, input_file, which_split):
        examples = []
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                next(f)
                for idx, line in enumerate(f):
                    segs = line.strip().split("\t")
                    if len(segs) < 4: 
                        continue
                    uid, passage, question, label = segs[0], segs[1], segs[2], segs[3]
                    examples.append(
                    SentencePairExample(uuid=uid, text_a=passage, text_b=question, label=label)
                    )
        except Exception as e:
            print(f"Couldn't print because of the following exception {e}")
        return examples


class MultiRCDataset(object):
    """
    MultiRC: each question has multiple candidate answers, each
    labeled 0 (wrong) or 1 (correct). Passage is text_a,
    question+candidate answer is text_b.
    TSV format: index \t passage \t question \t answer \t label
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev")
        self.tst_egs = self.get_split_examples("test")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, f"{which_split}.tsv")
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, input_file, which_split):
        examples = []
        with open(input_file, "r", encoding="utf-8") as f:
            next(f)
            for idx, line in enumerate(f):
                segs = line.strip().split("\t")
                uid, passage, question_answer, label = segs[0], segs[1], segs[2], segs[3]
                # Concatenate question and candidate answer as text_b
                examples.append(
                    SentencePairExample(uuid=uid, text_a=passage, text_b=question_answer, label=label)
                )
        return examples


class WSCDataset(object):
    """
    WSC: Winograd coreference. Given a sentence with a pronoun,
    determine if it refers to a given span.
    TSV format: index \t text \t span1 \t span2 \t label
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev")
        self.tst_egs = self.get_split_examples("test")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, f"{which_split}.tsv")
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, input_file, which_split):
        examples = []
        with open(input_file, "r", encoding="utf-8") as f:
            next(f)
            for idx, line in enumerate(f):
                segs = line.strip().split("\t")
                if len(segs) < 4:
                    continue
                uid, text, span1, label = segs[0], segs[1], segs[2], segs[3]
                examples.append(
                    SentencePairExample(uuid=uid, text_a=text, text_b=span1, label=label)
                )
        return examples


class COPADataset(object):
    """
    COPA: causal reasoning. Given a premise, pick the more
    plausible cause or effect from two alternatives.
    Framed as two binary examples per instance (one per alternative).
    TSV format: index \t premise \t choice1 \t choice2 \t question \t label
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev")
        self.tst_egs = self.get_split_examples("test")

    def get_labels(self):
        return ["0", "1"]

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, f"{which_split}.tsv")
        return self._create_examples(where_, which_split)

    def _create_examples(self, input_file, which_split):
        examples = []
        with open(input_file, "r", encoding="utf-8") as f:
            next(f)
            for idx, line in enumerate(f):
                segs = line.strip().split("\t")
                uid = f"{which_split}-{idx}"
                uid, premise, choice, label = segs[0], segs[1], segs[2], segs[3]
                # Encode as two separate premise+choice pairs, label is which choice is correct
                examples.append(
                    SentencePairExample(uuid=uid, text_a=premise, text_b=choice, label=label)
                )
        return examples