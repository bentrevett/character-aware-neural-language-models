import sys

from tqdm import tqdm

from torchtext import data
from torchtext import datasets

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import models

N_EPOCHS = 25
INIT_LR = 1.0
BATCH_SIZE = 20
SCHEDULER_PATIENCE = 0
SCHEDULER_FACTOR = 0.5
SCHEDULER_THRESHOLD = 1.0
CLIP = 5.0

EMBEDDING_DIM = 15
FILTER_SIZES = [1,2,3,4,5,6]
FILTER_CHANNELS = [filter_size * 25 for filter_size in FILTER_SIZES]
HIGHWAY_LAYERS = 1
HIDDEN_DIM = 300
N_LAYERS = 2

CHAR_NESTING = data.Field(batch_first=True, tokenize=list, init_token='<sos>', eos_token='<eos>')
CHARS = data.NestedField(CHAR_NESTING)
TARGET = data.Field(batch_first=True)

fields = {'words': ('chars', CHARS), 'target': ('target', TARGET)}

#get data from csv
train, test = data.TabularDataset.splits(
                path = 'data',
                train = 'ptb.valid.jsonl',
                #validation = 'ptb.valid.jsonl',
                test = 'ptb.valid.jsonl',
                format = 'json',
                fields = fields
)

print(dir(train))
print(train.fields)

TARGET.build_vocab(train)
CHARS.build_vocab(train)

print(f'{len(CHARS.vocab)} characters in character vocab')
print(f'char vocab = {CHARS.vocab.itos}')

print(f'{len(TARGET.vocab)} words in target vocab')
print(f'most common words = {TARGET.vocab.freqs.most_common(10)}')

train_iter, test_iter = data.Iterator.splits((train, test),
                                             batch_size=BATCH_SIZE,
                                             repeat=False)

model = models.CharLM(len(CHARS.vocab),
                      len(TARGET.vocab),
                      EMBEDDING_DIM,
                      FILTER_SIZES,
                      FILTER_CHANNELS,
                      HIGHWAY_LAYERS,
                      HIDDEN_DIM,
                      N_LAYERS)

print(model)

#initialize optimizer, scheduler and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=INIT_LR)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold=SCHEDULER_THRESHOLD, threshold_mode='abs', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, verbose=True)

#place on GPU
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

best_test_loss = float('inf')

for epoch in range(1, N_EPOCHS+1):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in tqdm(train_iter, desc='Train'):

        optimizer.zero_grad()

        predictions = model(batch.chars)

        assert 1 == 2

        loss = criterion(predictions, batch.target.squeeze(1))

        loss.backward()

        optimizer.step()

        pred = predictions.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        acc = pred.eq(batch.target.data.view_as(pred)).long().cpu().sum()

        epoch_loss += loss.data[0]
        epoch_acc += acc/len(pred)

    #calculate metrics averaged across whole batch
    train_loss = epoch_loss / len(train_iter)
    train_acc = epoch_acc / len(train_iter)

    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()

    for batch in tqdm(test_iter, desc=' Test'):

        predictions = model(batch.body, batch.prev)

        loss = criterion(predictions, batch.target.squeeze(1))
        
        pred = predictions.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        acc = pred.eq(batch.target.data.view_as(pred)).long().cpu().sum()

        epoch_loss += loss.data[0]
        epoch_acc += acc/len(pred)

    #calculate metrics averaged across whole epoch
    test_acc = epoch_acc / len(test_iter)
    test_loss = epoch_loss / len(test_iter)

    #print metrics
    print(f'Epoch: {epoch}') 
    print(f'Train Loss: {train_loss:.3f}, Train Acc.: {train_acc*100:.2f}%')
    print(f'Test Loss: {test_loss:.3f}, Test Acc.: {test_acc*100:.2f}%')

    if test_loss < best_test_loss:
        best_test_loss = test_loss