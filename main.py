import sys
import math

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
RNN_DIM = 300
RNN_LAYERS = 2
DROPOUT = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHAR_NESTING = data.Field(batch_first=True, tokenize=list, init_token='<sos>', eos_token='<eos>')
CHARS = data.NestedField(CHAR_NESTING)
TARGETS = data.Field(batch_first=True)

fields = {'words': ('chars', CHARS), 'targets': ('targets', TARGETS)}

#get data from csv
train, valid, test = data.TabularDataset.splits(
                path = 'data',
                train = 'ptb.train.jsonl',
                validation = 'ptb.valid.jsonl',
                test = 'ptb.test.jsonl',
                format = 'json',
                fields = fields
)

TARGETS.build_vocab(train)
CHARS.build_vocab(train)

print(f'{len(CHARS.vocab)} characters in character vocab')
print(f'char vocab = {CHARS.vocab.itos}')

print(f'{len(TARGETS.vocab)} words in target vocab')
print(f'most common words = {TARGETS.vocab.freqs.most_common(10)}')

train_iter, valid_iter, test_iter = data.Iterator.splits((train, valid, test),
                                             batch_size=BATCH_SIZE,
                                             sort=False,
                                             repeat=False)

model = models.CharLM(len(CHARS.vocab),
                      len(TARGETS.vocab),
                      EMBEDDING_DIM,
                      FILTER_SIZES,
                      FILTER_CHANNELS,
                      HIGHWAY_LAYERS,
                      RNN_DIM,
                      RNN_LAYERS,
                      DROPOUT)

print(model)

#initialize optimizer, scheduler and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=INIT_LR)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold=SCHEDULER_THRESHOLD, threshold_mode='abs', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, verbose=True)

#place on GPU
model.to(device)
criterion.to(device)

best_valid_loss = float('inf')

for epoch in range(1, N_EPOCHS+1):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in tqdm(train_iter, desc='Train'):

        optimizer.zero_grad()

        predictions = model(batch.chars)

        loss = criterion(predictions, batch.targets.view(-1,))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)

        optimizer.step()

        epoch_loss += loss.item()

    #calculate metrics averaged across whole epoch
    train_loss = epoch_loss / len(train_iter)

    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()

    for batch in tqdm(valid_iter, desc='Valid'):

        with torch.no_grad():
            
            predictions = model(batch.chars)

            loss = criterion(predictions, batch.targets.view(-1,))
            
            epoch_loss += loss.item()

    #calculate metrics averaged across whole epoch
    valid_loss = epoch_loss / len(valid_iter)

    #update learning rate
    scheduler.step(math.exp(valid_loss))

    #print metrics
    print(f'Epoch: {epoch}') 
    print(f'Train Loss: {train_loss:.3f}, Valid PPL: {math.exp(train_loss):.2f}')
    print(f'Valid Loss: {valid_loss:.3f}, Valid PPL: {math.exp(valid_loss):.2f}')

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')

model.load_state_dict(torch.load('model.pt'))
        
for batch in tqdm(test_iter, desc='Valid'):

        with torch.no_grad():

            predictions = model(batch.chars)

            loss = criterion(predictions, batch.targets.view(-1,))
            
            epoch_loss += loss.item()

#calculate metrics averaged across whole epoch
test_loss = epoch_loss / len(test_iter)

print(f'Test Loss: {test_loss:.3f}, Test PPL: {math.exp(test_loss):.2f}')