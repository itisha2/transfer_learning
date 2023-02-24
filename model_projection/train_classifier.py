from math import ceil
from random import seed, shuffle
import json
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_scheduler
from tqdm.auto import tqdm

# For shuffling
seed(42)

# For torch structures
torch.manual_seed(42)

MAJOR_POS = [
    'ADJ',
    'ADP',
    'ADV',
    'DET',
    'NOUN',
    'PRON',
    'VERB'
]


class ClassifierHead(nn.Module):
    def __init__(self, input_size=1024, hidden_size=2048):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        return self.linear2(x)


def prepare_sentence(sentence_dict):
    '''
    Create a copy of the input sentence for each token that
    has a POS tag from a pre-specified list.
    '''
    result = []
    for i, pos_tag in enumerate(sentence_dict['POS']):
        if pos_tag not in ['PUNCT', 'SCONJ', 'CCONJ', 'INTJ', 'X', 'SYM']:
            result.append({
                'sentence': [el for el in sentence_dict['tokens']],
                'offset': i,
                'POS': pos_tag,
                'label': sentence_dict['labels'][i]})
    return result


def get_embeddings_and_labels(batch, tokeniser, model):
    sentences = [el['sentence'] for el in batch]
    tokenisation = tokeniser(sentences, is_split_into_words=True,
                             padding=True, truncation=True, return_tensors='pt')
    model_inputs = {k: v.cuda() for k, v in tokenisation.items()}
    model_outputs = model(**model_inputs).last_hidden_state
    embeddings = []
    labels = []
    for i in range(len(batch)):
        # subword_embeddings = []
        for j, word_id in enumerate(tokenisation.word_ids(batch_index=i)):
            if word_id == batch[i]['offset']:
                embeddings.append(model_outputs[i, j])
                break
        labels.append(batch[i]['label'])
    assert len(embeddings) == len(labels)
    return torch.vstack(embeddings), torch.tensor(labels).cuda()


def train_epoch(epoch_n, train_data, batch_size,
                tokeniser, embedding_model, classifier_head,
                loss_function, scheduler, optimiser):
    embedding_model.train()
    epoch_losses = []
    n_training_steps = ceil(len(train_data) / batch_size)
    for step in tqdm(range(n_training_steps), desc=f'Train: epoch {epoch_n+1}', leave=False):
        optimiser.zero_grad()
        i = step * batch_size
        batch = train_data[i: i + batch_size]
        embeddings, labels = get_embeddings_and_labels(
            batch, tokeniser, embedding_model)
        logits = classifier_head(embeddings)
        loss = loss_function(logits, labels)
        # Do not call loss.item before stepping!
        loss.backward()
        optimiser.step()
        scheduler.step()
        epoch_losses.append(loss.item())
    return torch.tensor(epoch_losses).mean().item()


def validate(epoch_n, dev_data, batch_size,
             tokeniser, embedding_model, classifier_head):
    embedding_model.eval()
    n_validation_steps = ceil(len(dev_data) / batch_size)
    all_gold = []
    all_pred = []
    for step in tqdm(range(n_validation_steps), desc=f'Validation: epoch {epoch_n+1}', leave=False):
        i = step * batch_size
        batch = dev_data[i: i + batch_size]
        with torch.no_grad():
            embeddings, gold_labels = get_embeddings_and_labels(
                batch, tokeniser, embedding_model)
            all_gold.extend(list(gold_labels.cpu().numpy()))
            logits = classifier_head(embeddings)
            all_pred.extend(list(logits.argmax(dim=1).cpu().numpy()))
    return f1_score(all_gold, all_pred)


with open('../json/amsterdam_POS_tagged_train.json') as inp:
    data = json.load(inp)
prepared_data = []
for sentence_dict in tqdm(data, desc='Preparing data', leave=False):
    prepared_data.extend(prepare_sentence(sentence_dict))

# When splitting by token:
# shuffle(prepared_data)
# cutoff = len(prepared_data) // 10
# train_data_pos = [el for el in prepared_data[cutoff:] if el['label'] == 1]
# train_data_neg = [el for el in prepared_data[cutoff:] if el['label'] == 0]

train_data_pos = [el for el in prepared_data if el['label'] == 1]
train_data_neg = [el for el in prepared_data if el['label'] == 0]

with open('../json/amsterdam_POS_tagged_test.json') as inp:
    dev_data = json.load(inp)
prepared_data_dev = []
for sentence_dict in tqdm(dev_data, desc='Preparing dev data', leave=False):
    prepared_data_dev.extend(prepare_sentence(sentence_dict))
# dev_data_pos = [el for el in prepared_data[:cutoff] if el['label'] == 1]
# dev_data_neg = [el for el in prepared_data[:cutoff] if el['label'] == 0]
# Balance the dev data
dev_data_pos = [el for el in prepared_data_dev if el['label'] == 1]
dev_data_neg = [el for el in prepared_data_dev if el['label'] == 0]
shuffle(dev_data_neg)
dev_data = dev_data_pos + dev_data_neg[:len(dev_data_pos)]

model_name = 'bert-large-uncased'
tokeniser = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name)
embedding_model.cuda()
embedding_model = nn.DataParallel(embedding_model)
classifier_head = ClassifierHead()
classifier_head.cuda()
classifier_head = nn.DataParallel(classifier_head)
optimiser = AdamW(list(embedding_model.parameters()) +
                  list(classifier_head.parameters()), lr=1e-5)
n_epochs = 20
batch_size = 64 * 7
n_training_steps = ceil(len(train_data_pos) / batch_size) * n_epochs
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimiser,
    num_warmup_steps=0,
    num_training_steps=n_training_steps
)
loss_function = nn.CrossEntropyLoss()

for epoch_n in range(n_epochs):
    shuffle(train_data_neg)
    train_data = train_data_neg[:len(train_data_pos)] + train_data_pos
    shuffle(train_data)
    epoch_loss = train_epoch(epoch_n, train_data, batch_size, tokeniser,
                             embedding_model, classifier_head,
                             loss_function, lr_scheduler, optimiser)
    print(f'Epoch {epoch_n+1} train loss: {epoch_loss}')

    # Validate
    f1 = validate(epoch_n, dev_data, batch_size * 2,
                  tokeniser, embedding_model, classifier_head)
    print(f'Epoch {epoch_n+1} validation F1: {f1}')
    print()
