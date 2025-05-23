import importlib
from tqdm import tqdm
import pandas as pd
import spacy
import pickle
from collections import defaultdict, Counter
from dataset.utils import load_dataset_file

import fasttext.util

language_source = "de_dep_news_trf"

def get_parts_of_speech(sentence):
    doc = nlp(sentence)
    normalized_sentence = [token.lemma_ for token in doc]
    pos_tags = [(token.text, token.pos_) for token in doc]
    return [(n.lower(), p[0],p[1]) for n, p in zip(normalized_sentence, pos_tags)]

pkl_dir = [f"data/labels.{x}" for x in ["train", "dev", "test"]]

dict_pos = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary verb",
    "CONJ": "conjunction",
    "CCONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
}

selected_vocab = ["NOUN", "NUM", "ADV", "PRON", "PROPN", "ADJ", "VERB"]

nlp = spacy.load(language_source)

translations = []

for pkl in pkl_dir:
    raw_data = load_dataset_file(pkl)
    list_of_keys = [key for key,value in raw_data.items()]
    for idx in range(len(list_of_keys)):
        key = list_of_keys[idx]
        tgt_sample = raw_data[key]['text']
        translations.append(tgt_sample)
    print(f'{pkl} Loaded!')


dict_sentence = {}
all_lens = []
for sentence in tqdm(translations):
    pos = get_parts_of_speech(sentence)
    lems = []
    for lem, word, part in pos:
        if part in selected_vocab:
            lems.append(lem)
    all_lens.extend(lems)
    dict_sentence[sentence] = lems

fasttext.util.download_model('de', if_exists='ignore') 
ft = fasttext.load_model('cc.de.300.bin')

dict_lem_to_id = {l:i for i, l in enumerate(list(set(all_lens)))}


# flat_list = [item for sublist in list(dict_sentence.values()) for item in sublist]
# count = Counter(flat_list)

# vector = torch.zeros((len(dict_lem_to_id),300))
# for key, value in tqdm(dict_lem_to_id.items()):
#     vector[value] = torch.tensor(ft.get_word_vector(key))


dict_processed_words = {
    "dict_lem_counter": dict(Counter(all_lens)),
    "dict_sentence": dict(dict_sentence),
    "dict_lem_to_id": dict_lem_to_id
}

with open("data/processed_words.phx_pkl", "wb") as f:
    pickle.dump(dict_processed_words, f)

    