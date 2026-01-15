"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from torchtext.data import Field, BucketIterator, Example, Dataset
from datasets import load_dataset


class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')

    def make_dataset(self):
        if self.ext == ('.de', '.en'):
            self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        elif self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
        dataset = load_dataset("bentrevett/multi30k")
        src_lang, trg_lang = self.ext
        def convert_to_torchtext(hf_dataset):
            fields = [('src', self.source), ('trg', self.target)]
            examples = [
                Example.fromlist([ex['en' if src_lang == 'en' else 'de'],
                                  ex['de' if trg_lang == 'de' else 'en']],
                                 fields)
                for ex in hf_dataset
            ]
            return Dataset(examples, fields)

        train_data = convert_to_torchtext(dataset['train'])
        valid_data = convert_to_torchtext(dataset['validation'])
        test_data = convert_to_torchtext(dataset['test'])
        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train, validate, test),
            batch_size=batch_size,
            device=device,
            sort_key=lambda x: len(x.src),
            sort_within_batch=True
        )
        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator