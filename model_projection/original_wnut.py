from datasets import load_dataset
from transformers import AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_and_align_labels(examples):
    # examples is a batch of input data
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    #print(type(examples)) # <class 'datasets.arrow_dataset.Batch'>
    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        print("iy....", i, label)
        
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
    #print("iy....", tokenized_inputs)
  
    return tokenized_inputs

if __name__ == "__main__":
    
    wnut = load_dataset("wnut_17")
 

    #print("Printing NER tags : ")
    label_list = wnut["train"].features[f"ner_tags"].feature.names
    #print(label_list)


    tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)
    print(tokenized_wnut)