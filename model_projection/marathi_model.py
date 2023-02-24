from create_data import *
from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd
import datasets
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import TFAutoModelForTokenClassification



tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
data_collator = DataCollatorForTokenClassification(tokenizer)


def tokenize_and_align_labels(examples):
    # examples is a batch of input data
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[f"pos_tags"]):   
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:                            # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:              # Only label the first token of a given word.
                label_ids.append(label[word_idx])

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def generate_dataset_arrow(wnut):
    # https://huggingface.co/docs/datasets/about_arrow

    """
    1) Arrow enables large amounts of data to be processed and moved quickly.
    2) Arrows stores data in columnar memory layout.
    3) Arrow is column-oriented so it is faster at columns of data.
    4) Arrow allows for copy-free hand-offs to standard machine learning tools such as NumPy, Pandas, PyTorch, and TensorFlow.
    """

    # see the below sample code
    #dataset = Dataset.from_dict({"a": [0, 1, 2]})
    #dataset.map(lambda batch: {"b": batch["a"] * 2}, batched=True)  # new column with 6 elements: [0, 1, 2, 0, 1, 2]   

    wnut_dataset = {}
    wnut_train_df = pd.DataFrame.from_records(wnut['train'])
    wnut_dataset['train'] = Dataset.from_dict(wnut_train_df)
    wnut_dev_df = pd.DataFrame.from_records(wnut['dev'])
    wnut_dataset['dev'] = Dataset.from_dict(wnut_dev_df)
    wnut_test_df = pd.DataFrame.from_records(wnut['test'])
    wnut_dataset['test'] = Dataset.from_dict(wnut_test_df)
    # look at the error for from_dict function, https://huggingface.co/docs/datasets/about_map_batch
    wnut_dataset = datasets.DatasetDict(wnut_dataset) 
    return wnut_dataset


def main():
    train_path = './marathi_data/UD_Marathi-UFAL/mr_ufal-ud-train.conllu'
    dev_path = './marathi_data/UD_Marathi-UFAL/mr_ufal-ud-dev.conllu'
    test_path = './marathi_data/UD_Marathi-UFAL/mr_ufal-ud-test.conllu'

    wnut = createData(train_path, dev_path, test_path)

    wnut_dataset = generate_dataset_arrow(wnut)
    
   
    
    """
    print(wnut_dataset)
    print("****************************************")
    pos_list = []
    k = 0
    for i in wnut_dataset["train"]:
        k += 1
        #print(k, i)
        #print()
        pos_list.append(i["pos_tags"])
    tokenize_and_align_labels(wnut_dataset["train"][0])

    """
    # run it using the map function
    # Dataset from hugging face: Backed by the Apache Arrow format,
    # for complete dataset

   # goal is to generate Dataset Arrow:
   #https://huggingface.co/docs/datasets/about_arrow

    tokenized_wnut = wnut_dataset.map(tokenize_and_align_labels, batched=True)

    tf_train_set = tokenized_wnut["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
    )

    tf_validation_set = tokenized_wnut["validation"].to_tf_dataset(
        columns=["attention_mask", "input_ids", "labels"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )
    model = TFAutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_list))
    


if __name__ == "__main__":
    main()


"""
Steps followed:
1) used conllu instead of pyconll, to replace None with "_".
2) converted wnut to dataset format provided by huggingface.



I looked into the error you got. It seems that one problem is that the code in the repository expect wnut to be a Dataset, which is a special HuggingFace data structure. It is special because when you ask for wnut['train'][0], it gives you

{'id': '0',
'pos_tags': ['DET', 'AUX', 'NOUN', 'PUNCT'],
'tokens': ['एक', 'असणे', 'राजा', '.']}

but when you ask for wnut['train']['pos_tags'], it gives you an array of POS tags for all sentences, so

for i, label in enumerate(examples[f"pos_tags"])

iterates over POS tags for the first sentence, POS tags for the second sentence, etc., and label is a list.

Yes, exactly. The task is to balance the realistic look of the sentence with ease of analysis. The tokens from the list is what mBERT ends up seeing, so if it sees a sentence looking like "परंतु_लाडमुळेतोबिघडणे", it'll get confused because it hasn't seen this in training. Unfortunately, if it sees a sentence with contracted forms replaced with their components (we're basically replacing "isn't" with "is + not"), it may also get confused, because it hasn't seen this in training either.

A possible workaround is, instead of replacing contracted forms with their components and using POS tags of the components, to use the original contracted form and give it another POS tag (e.g., X). I guess contracted forms are found both in Hindi and Marathi, so this can be used for transfer, but contracted forms for English are pre-split in UD ("isn't" is represented as "is + n't"), so there will be a mismatch in the set of POS tags.




Next Steps:

1) The code works for first interation, check y is it failing for the second iteration.
2) Find out, y do we have to send of the [list of [list of pos tags]] for all sentences everytime?
"""