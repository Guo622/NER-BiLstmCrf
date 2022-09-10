import torch
import torchtext
import random
import logging
import numpy as np
from config import Config

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_path, name='train.log', mode='a'):
    logging.basicConfig(
        level=logging.INFO,
        filename=log_path+name,
        filemode=mode,
        format='%(asctime)s-%(message)s'
    )


def build_optimizers(config: Config, model: torch.nn.Module, max_steps, use_scheduler = True):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay
    )
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_steps, eta_min=config.min_lr)
    else:
        scheduler = None

    return optimizer, scheduler

def load_pretrain_embeddings(model, vocab, cache_dir=None):
    vectors = torchtext.vocab.GloVe(name='840B', dim=300, cache=cache_dir)
    embeddings = vectors.get_vecs_by_tokens(vocab.get_itos())
    model.word_embedding.weight.data = embeddings
