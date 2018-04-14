from torchtext import data
from torchtext import datasets
import re
import utils

def char_tokenizer(words):
    return list(' '.join(words))

WORDS = data.Field()
CHARS = data.Field(preprocessing=char_tokenizer)

train, valid, test = utils.PennTreebankChar.splits(WORDS, CHARS)

print(dir(train))
print(train.fields)
#print(test.fields)
#print(valid.fields)

WORDS.build_vocab(train)
CHARS.build_vocab(train)

print(f'{len(WORDS.vocab)} words in words vocab')
print(f'most common words = {WORDS.vocab.freqs.most_common(10)}')

print(f'{len(CHARS.vocab)} characters in character vocab')



train_iter, valid_iter, test_iter = utils.BPTTIteratorChar.splits((train, valid, test),
                                                             batch_size=32,
                                                             bptt_len=30, # this is where we specify the sequence length
                                                             repeat=False)

batch = next(iter(train_iter))

print(dir(batch))

print(batch.text.shape)
print(batch.chars.shape)

example = list(batch.text[0].data)
text = [WORDS.vocab.itos[i] for i in example]
print(text)

example = list(batch.chars[0][0].data)
text = [CHARS.vocab.itos[i] for i in example]
print(text)

example = list(batch.chars[0][1].data)
text = [CHARS.vocab.itos[i] for i in example]
print(text)

