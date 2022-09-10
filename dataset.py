import torch
import datasets
import torchtext
import functools
from config import Config
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

SPECIAL_TOKENS = ['<unk>', '<pad>']

TAG2IDX = {
    'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3,
    'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7,
    'I-MISC': 8, '<beg>': 9, '<end>': 10
}

START_TAG = '<beg>'
END_TAG = '<end>'

START_TAG_IDX = TAG2IDX[START_TAG]
END_TAG_IDX = TAG2IDX[END_TAG]

def build_dataloaders(config: Config):
    train_data, valid_data, test_data = datasets.load.load_dataset(
        'conll2003', cache_dir=config.cache_dir, split=['train', 'validation', 'test']
    )
    vocab = torchtext.vocab.build_vocab_from_iterator(
        train_data['tokens']+valid_data['tokens']+test_data['tokens'],
        specials=SPECIAL_TOKENS
    )
    unk_idx = vocab['<unk>']
    pad_idx = vocab['<pad>']
    vocab.set_default_index(unk_idx)

    def preprocess(data):
        def func(example, vocab):
            return {
                'ids': [vocab[token] for token in example['tokens']],
                'length': len(example['tokens'])
            }
        data = data.map(func, fn_kwargs={'vocab': vocab})
        data = data.with_format(type='torch', columns=['ids', 'ner_tags', 'length'])
        return data

    train_data = preprocess(train_data)
    valid_data = preprocess(valid_data)
    test_data = preprocess(test_data)

    collate_fn_ = functools.partial(collate_fn, pad_idx=pad_idx, device=config.device)
    train_loader = DataLoader(
        train_data, batch_size=config.train_batch_size, collate_fn=collate_fn_, shuffle=True)
    valid_loader = DataLoader(
        valid_data, batch_size=config.val_batch_size, collate_fn=collate_fn_)
    test_loader = DataLoader(
        test_data, batch_size=config.test_batch_size, collate_fn=collate_fn_)

    config.vocab_size = len(vocab)
    return train_loader, valid_loader, test_loader, vocab

def collate_fn(batch, pad_idx, device):
    batch_ids = [sample['ids'] for sample in batch]
    batch_ids = pad_sequence(batch_ids,batch_first=True,padding_value=pad_idx)
    batch_label = [sample['ner_tags'] for sample in batch]
    batch_label = pad_sequence(batch_label,batch_first=True,padding_value=-1)
    mask = [torch.ones((sample['length'].item())) for sample in batch]
    mask = pad_sequence(mask, batch_first=True, padding_value=0)
    return {
        'ids': batch_ids.long().to(device),
        'label': batch_label.long().to(device),
        'lengths': mask.sum(1).long().cpu(),
        'mask': mask.long().to(device)
    }  