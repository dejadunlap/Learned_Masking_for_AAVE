import my_regex
import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")
detector = my_regex.LinguisticFeatureDetector(nlp=nlp)

def is_aave(sent):
    aint = detector.has_aint(sent) 
    be = detector.has_habitual_be(sent)
    comp = detector.has_double_comparative(sent)
    multi_mods = detector.has_multiple_modals(sent)
    neg_concord = detector.has_negative_concord(sent)
    done = detector.has_perfective_done(sent)

    if aint or be or comp or multi_mods or neg_concord or done: 
        return True
    
    return False

# Load a local CSV file
df = pd.read_csv('data/aave_data/test.tsv', sep='\t')
golds = df["label"]
pred = [is_aave(x) for x in df["sentence"]]

correct = sum(1 for x, y in zip(golds, pred) if x == y)
accuracy = correct / len(df["label"])
print(f"Accuracy of Regex is {accuracy}")