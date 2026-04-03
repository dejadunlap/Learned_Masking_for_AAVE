import os
from .common import SentencePairExample, SentenceExample

__all__ = [
    "AAVEDataset",
    "QNLIDataset",
    "SST2Dataset",
    "RTEDataset",
    "MNLIDataset",
    "QQPDataset", 
    "COLADataset",
    "WNLIDataset",
    "STSBDataset",
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

class QNLIDataset(GlueDataset):
    """ a sentence pair dataset converted from squad. """

    def __init__(self, data_dir):
        self.name = "qnli"
        self.data_dir = data_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev")
        self.tst_egs = self.get_split_examples("test")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.tsv".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_examples(self, input_file, which_split):
        """ parse and convert raw string to SentencePairExample """
        sentence_pair_egs = []
        with open(input_file, "r") as f:
            next(f)
            for idx, line in enumerate(f):
                segs = line.strip().split("\t")
                assert len(segs) == 4
                label = segs[-1]
                text_a, text_b = segs[1], segs[2]
                uid = "%s-%s" % (which_split, idx)
                sentence_pair_egs.append(
                    SentencePairExample(
                        uid=uid, text_a=text_a, text_b=text_b, label=label
                    )
                )
        return sentence_pair_egs

class RTEDataset(GlueDataset):
    """ a sentence pair dataset converted from squad. """

    def __init__(self, data_dir):
        self.name = "rte"
        self.data_dir = data_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev")
        self.tst_egs = self.get_split_examples("test")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.tsv".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_examples(self, input_file, which_split):
        """ parse and convert raw string to SentencePairExample """
        sentence_pair_egs = []
        with open(input_file, "r") as f:
            next(f)
            for idx, line in enumerate(f):
                segs = line.strip().split("\t")
                assert len(segs) == 4
                label = segs[-1]
                text_a, text_b = segs[1], segs[2]
                uid = "%s-%s" % (which_split, idx)
                sentence_pair_egs.append(
                    SentencePairExample(
                        uid=uid, text_a=text_a, text_b=text_b, label=label
                    )
                )
        return sentence_pair_egs

class MNLIDataset(GlueDataset):
    """ 
    multi-genre NLI: 
        for a pair of sentences, predict whether the second sentence is an entailment, 
        contradiction, or neutral w.r.t the first one. 
    NB: 
        this dataset seems to be problematic.
        BERT only use MNLI-m, in domain classification.
    """

    def __init__(self, data_dir):
        self.name = "mnli"
        self.data_dir = data_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev_matched")
        self.tst_egs = self.get_split_examples("test")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.tsv".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, input_file, which_split):
        """ parse and convert raw string to SentenceExample """
        sentence_pair_egs = []
        with open(input_file, "r", encoding="utf-8") as f:
            next(f)
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                uid = "%s-%s" % (which_split, idx)
                text_a, text_b = line[8], line[9]
                label = line[-1]
                sentence_pair_egs.append(
                    SentencePairExample(
                        uid=uid, text_a=text_a, text_b=text_b, label=label
                    )
                )
        return sentence_pair_egs


class QQPDataset(GlueDataset):
    """
    Determine whether a pair of questions are semantically equivalent.
    """

    def __init__(self, data_dir):
        self.name = "qqp"
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
        """ parse and convert raw string to SentenceExample 
        there are some ill examples in this dataset so they are skipped.
        """
        sentence_pair_egs = []
        with open(input_file, "r", encoding="utf-8") as f:
            next(f)
            for idx, line in enumerate(f):
                line = line.strip().split("\t")
                uid = "%s-%s" % (which_split, idx)
                try:
                    text_a, text_b, label = line[3], line[4], line[5]
                except IndexError:  # it seems transformers also did this
                    print(idx, line)
                sentence_pair_egs.append(
                    SentencePairExample(
                        uid=uid, text_a=text_a, text_b=text_b, label=label
                    )
                )
        return sentence_pair_egs

class COLADataset(GlueDataset):
    def __init__(self, data_dir):
        self.name = "cola"
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
                text_a, label = line[3], line[1]
                uid = "%s-%s" % (which_split, idx)
                sentence_egs.append(
                    SentenceExample(uid=uid, text_a=text_a, label=label)
                )
        return sentence_egs

class WNLIDataset(GlueDataset):
    """ a sentence pair dataset converted from squad. """

    def __init__(self, data_dir):
        self.name = "qnli"
        self.data_dir = data_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev")
        self.tst_egs = self.get_split_examples("test")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.tsv".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        return self._create_examples(where_, which_split)

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_examples(self, input_file, which_split):
        """ parse and convert raw string to SentencePairExample """
        sentence_pair_egs = []
        with open(input_file, "r") as f:
            next(f)
            for idx, line in enumerate(f):
                segs = line.strip().split("\t")
                assert len(segs) == 4
                label = segs[-1]
                text_a, text_b = segs[1], segs[2]
                uid = "%s-%s" % (which_split, idx)
                sentence_pair_egs.append(
                    SentencePairExample(
                        uid=uid, text_a=text_a, text_b=text_b, label=label
                    )
                )
        return sentence_pair_egs
    
class STSBDataset(GlueDataset):
    """ a sentence pair dataset converted from squad. """

    def __init__(self, data_dir):
        self.name = "stsb"
        self.data_dir = data_dir
        self.trn_egs = self.get_split_examples("train")
        self.val_egs = self.get_split_examples("dev")
        self.tst_egs = self.get_split_examples("test")

    def get_split_examples(self, which_split):
        where_ = os.path.join(self.data_dir, "{}.tsv".format(which_split))
        print("[INFO] {} is looking for {}".format(self.__class__.__name__, where_))
        exs= self._create_examples(where_, which_split)
        return exs

    def _create_examples(self, input_file, which_split):
        """ parse and convert raw string to SentencePairExample """
        sentence_pair_egs = []
        with open(input_file, "r") as f:
            next(f)
            for idx, line in enumerate(f):
                segs = line.strip().split("\t")
                assert len(segs) == 4
                label = segs[-1]
                text_a, text_b = segs[1], segs[2]
                uid = "%s-%s" % (which_split, idx)
                sentence_pair_egs.append(
                    SentencePairExample(
                        uid=uid, text_a=text_a, text_b=text_b, label=label
                    )
                )
        return sentence_pair_egs