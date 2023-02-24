# This snippet finds sentences where a token marked with part of speech 'AUX' are
# governed by a NOUN. For example, in French this is a less common construction
# and we may want to validate these examples because we have previously found some
# problematic examples of this construction.

"""
Author: Itisha Yadav
Date: 6/08/2022
Place: Stuttgart
"""
import pyconll
from sklearn import preprocessing
from conllu import parse

def demo():
    data_file = open("./demo.conllu", "r", encoding="utf-8").read()
    train = parse(data_file)
    train_data = []
    unique_tags = set()
    k = 0
    # Conll objects are iterable over their sentences, and sentences are iterable
    # over their tokens. Sentences also de/serialize comment information.
    for sentence in train:
        k += 1
        pos_tags = []
        tokens = []
        for token in sentence:
            if isinstance(token["id"], int):
                pos_tags.append(token["upos"])
                tokens.append(token["lemma"])
                unique_tags.add(token["upos"])
            """
            elif isinstance(token["id"], tuple):
                pos_tags.append("X")
                tokens.append(token["form"])
                unique_tags.add("X")
            """
        #temp = {"id": sentence.metadata["sent_id"], "pos_tags": pos_tags, "tokens": tokens}
        temp = {"id": k, "pos_tags": pos_tags, "tokens": tokens}
        print(temp)


def createDataUtility(data_path):
    #train = pyconll.load_from_file(data_path)
    data_file = open(data_path, "r", encoding="utf-8").read()
    train = parse(data_file)
    train_data = []
    unique_tags = set()
    k = 0
    # Conll objects are iterable over their sentences, and sentences are iterable
    # over their tokens. Sentences also de/serialize comment information.
    for sentence in train:
        k += 1
        pos_tags = []
        tokens = []
        for token in sentence:
            if isinstance(token["id"], int):
                pos_tags.append(token["upos"])
                tokens.append(token["lemma"])
                unique_tags.add(token["upos"])
        #temp = {"id": sentence.metadata["sent_id"], "pos_tags": pos_tags, "tokens": tokens}
        temp = {"id": k, "pos_tags": pos_tags, "tokens": tokens}
        print(temp)
        
        train_data.append(temp)

    return train_data, list(unique_tags)


"""
def createDataUtility(data_path):
    train = pyconll.load_from_file(data_path)
    train_data = []
    unique_tags = set()
    # Conll objects are iterable over their sentences, and sentences are iterable
    # over their tokens. Sentences also de/serialize comment information.
    for sentence in train:
        pos_tags = []
        tokens = []
        for token in sentence:
            pos_tags.append(token.upos)
            tokens.append(token.lemma)
            unique_tags.add(token.upos)
        temp = {"id": sentence.id, "pos_tags": pos_tags, "tokens": tokens}
        train_data.append(temp)

    return train_data, list(unique_tags)

"""

def encode_pos_labels(data, label_mapping):
    modified_data = []
    for i in data:
        i["pos_tags"] = [label_mapping[i] for i in i["pos_tags"]]
        modified_data.append(i)
    return modified_data


def createData(train_data_path, dev_data_path, test_data_path):
    wnut_train, utags_train = createDataUtility(train_data_path)
    wnut_dev, utags_dev = createDataUtility(dev_data_path)
    wnut_test, utags_test = createDataUtility(test_data_path)
    utags = list(set(utags_train + utags_dev + utags_test))
    print("Shape of train data : ", len(wnut_train))
    print("Shape of dev data : ", len(wnut_dev))
    print("Shape of test data : ", len(wnut_test))
    utags = [i for i in utags if i]
    print("Total Unique Tags = ", len(utags))
    label_list = ["O"] + utags
    label_list_mapping = {}
    for ind, i in enumerate(label_list):
        label_list_mapping[i] = ind
    print("Total number of classification token tags = ", len(label_list))
    return {"train": encode_pos_labels(wnut_train, label_list_mapping), "dev": encode_pos_labels(wnut_dev, label_list_mapping), 
    "test": encode_pos_labels(wnut_test, label_list_mapping)}


if __name__ == "__main__":
    train_path = './marathi_data/UD_Marathi-UFAL/mr_ufal-ud-train.conllu'
    dev_path = './marathi_data/UD_Marathi-UFAL/mr_ufal-ud-dev.conllu'
    test_path = './marathi_data/UD_Marathi-UFAL/mr_ufal-ud-test.conllu'

    #wnut, label_list = createData(train_path, dev_path, test_path)
    #print(label_list)
    demo()
