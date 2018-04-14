import io
import torch
import torchtext.data as data
from torch.autograd import Variable
from torchtext.data.iterator import Iterator
from torchtext.data.dataset import Dataset
from torchtext.data.batch import Batch
import math
import torch.nn.functional as F

class LanguageModelingDatasetChar(data.Dataset):
    """Defines a dataset for language modeling."""

    def __init__(self, path, text_field, char_field, newline_eos=True,
                 encoding='utf-8', **kwargs):
        """Create a LanguageModelingDataset given a path and a field.
        Arguments:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            newline_eos: Whether to add an <eos> token for every newline in the
                data file. Default: True.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [(('text', 'chars'), (text_field, char_field))]
        words = []
        chars = []
        with io.open(path, encoding=encoding) as f:
            for line in f:
                words += text_field.preprocess(line)
                chars += char_field.preprocess(line)
                if newline_eos:
                    words.append(u'<eos>')

        examples = [data.Example.fromlist([words], fields)]

        super(LanguageModelingDatasetChar, self).__init__(
            examples, fields, **kwargs)

class PennTreebankChar(LanguageModelingDatasetChar):
    """The Penn Treebank dataset.
    A relatively small dataset originally created for POS tagging.
    References
    ----------
    Marcus, Mitchell P., Marcinkiewicz, Mary Ann & Santorini, Beatrice (1993).
    Building a Large Annotated Corpus of English: The Penn Treebank
    """

    urls = ['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt',
            'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt']
    name = 'penn-treebank'
    dirname = ''

    @classmethod
    def splits(cls, text_field, char_field, root='.data', train='ptb.train.txt',
               validation='ptb.valid.txt', test='ptb.test.txt',
               **kwargs):
        """Create dataset objects for splits of the Penn Treebank dataset.
        Arguments:
            text_field: The field that will be used for text data.
            root: The root directory where the data files will be stored.
            train: The filename of the train data. Default: 'ptb.train.txt'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'ptb.valid.txt'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'ptb.test.txt'.
        """

        return super(PennTreebankChar, cls).splits(
            root=root, train=train, validation=validation, test=test,
            text_field=text_field, char_field=char_field, **kwargs)

def stack_char_data(data, start, end, char_vocab):
    data = data[start:end]
    max_word_len = max([d.shape[0] for d in data])
    pad_idx = char_vocab.vocab.stoi['<pad>']
    for i, d in enumerate(data):
        if d.shape[0] < max_word_len:
            padding = Variable(torch.LongTensor(max_word_len-d.shape[0],1).fill_(pad_idx).cuda())
            data[i] = torch.cat((d, padding), dim=0)
    print(len(data))
    print(torch.stack(data))
    return torch.stack(data)


class BPTTIteratorChar(Iterator):
    """Defines an iterator for language modeling tasks that use BPTT.
    Provides contiguous streams of examples together with targets that are
    one timestep further forward, for language modeling training with
    backpropagation through time (BPTT). Expects a Dataset with a single
    example and a single field called 'text' and produces Batches with text and
    target attributes.
    Attributes:
        dataset: The Dataset object to load Examples from.
        batch_size: Batch size.
        bptt_len: Length of sequences for backpropagation through time.
        sort_key: A key to use for sorting examples in order to batch together
            examples with similar lengths and minimize padding. The sort_key
            provided to the Iterator constructor overrides the sort_key
            attribute of the Dataset, or defers to it if None.
        train: Whether the iterator represents a train set.
        repeat: Whether to repeat the iterator for multiple epochs.
        shuffle: Whether to shuffle examples between epochs.
        sort: Whether to sort examples according to self.sort_key.
            Note that repeat, shuffle, and sort default to train, train, and
            (not train).
        device: Device to create batches on. Use -1 for CPU and None for the
            currently active GPU device.
    """

    def __init__(self, dataset, batch_size, bptt_len, **kwargs):
        self.bptt_len = bptt_len
        super(BPTTIteratorChar, self).__init__(dataset, batch_size, **kwargs)

    def __len__(self):
        return math.ceil((len(self.dataset[0].text) / self.batch_size - 1) /
                         self.bptt_len)

    def __iter__(self):
        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        CHARS = self.dataset.fields['chars']
        TEXT.eos_token = None
        text = text + ([TEXT.pad_token] * int(math.ceil(len(text) / self.batch_size) *
                                              self.batch_size - len(text)))
        data = TEXT.numericalize(
            [text], device=self.device, train=self.train)

        char_data = [CHARS.numericalize([t], device=self.device, train=self.train) for t in text]

        max_chars = max([c.shape[0] for c in char_data])

        pad_idx = CHARS.vocab.stoi['<pad>']

        for i, c in enumerate(char_data):
            if c.shape[0] < max_chars:
                padding = Variable(torch.LongTensor(max_chars-c.shape[0],1).fill_(pad_idx).cuda())
                char_data[i] = torch.cat((c, padding), dim=0)
        
        char_data = torch.stack(char_data)

        print(char_data.shape)

        char_data = char_data.view(self.batch_size, -1, max_chars).permute(1, 0, 2).contiguous()

        print(char_data.shape)

        data = data.view(self.batch_size, -1).t().contiguous()

        dataset = Dataset(examples=self.dataset.examples, fields=[
            ('text', TEXT), ('chars', CHARS), ('target', TEXT)])
        
        while True:
            for i in range(0, len(self) * self.bptt_len, self.bptt_len):
                self.iterations += 1
                seq_len = min(self.bptt_len, len(data) - i - 1)
                yield Batch.fromvars(
                    dataset, self.batch_size, train=self.train,
                    text=data[i:i + seq_len],
                    chars=char_data[i:i+seq_len],
                    target=data[i + 1:i + 1 + seq_len])
            if not self.repeat:
                return