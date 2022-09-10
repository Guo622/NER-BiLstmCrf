import os
import time
import torch
import logging
from tqdm import tqdm
from config import Config
from model import BiLstmCrf
from dataset import build_dataloaders
from sklearn.metrics import f1_score
from util import setup_logging, setup_seed, build_optimizers, load_pretrain_embeddings


def validate(model, val_loader):
    model.eval()
    labels = []
    preds = []
    tqdm_loader = tqdm(val_loader)
    tqdm_loader.set_description("validate...")
    with torch.no_grad():
        for batch in tqdm_loader:
            ids, label, length, mask = batch['ids'], batch['label'], batch['lengths'], batch['mask']
            emission = model(ids, length)
            paths = model.viterbi_decode(emission, length, mask)
            for p in paths:
                preds.extend(p)
            for b in range(label.shape[0]):
                labels.extend(label[b][:length[b]].cpu().numpy())
                
    f1_micro = f1_score(labels, preds, average='micro')
    model.train()
    return f1_micro


def train_and_validate(config:Config):
    train_loader, val_loader, _, vocab = build_dataloaders(config)
    max_steps = len(train_loader)*config.max_epochs
    model = BiLstmCrf(config)
    load_pretrain_embeddings(model, vocab, cache_dir=config.cache_dir)
    if config.checkpoint is not None:
        model.load_state_dict(torch.load(config.checkpoint, map_location='cpu'), strict=False)

    model.to(config.device)
    optimizer, scheduler = build_optimizers(config, model, max_steps, use_scheduler=True)

    step = 0
    start_time = time.time()

    for epoch in range(config.max_epochs):
        tqdm_loader = tqdm(train_loader)
        for i, batch in enumerate(tqdm_loader):
            ids, label, length, mask = batch['ids'], batch['label'], batch['lengths'], batch['mask']
            model.train()
            emission = model(ids, length)            
            loss = model.neg_log_likelihood(emission,label,length,mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            step += 1
            if step % config.print_steps == 0:
                time_per_step = (time.time() - start_time) / step
                remaining_time = time_per_step * (max_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}")
            tqdm_loader.set_description(f"epoch {epoch} bacth {i} loss:{loss:.3f}")
        
        f1_micro = validate(model, val_loader)
        logging.info(f"Epoch {epoch} valid: f1_micro {f1_micro:.3f}\n")
        print(f"valid: f1_micro {f1_micro:.3f}\n")
        
        torch.save(model.state_dict(), f"{config.save_dir}/epoch{epoch}_{f1_micro:.3f}.pth")


if __name__ == '__main__':

    config = Config()
    
    # config.load_from_dict(f'./config/config.json') # 复现用

    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.cache_dir, exist_ok=True)
    os.makedirs(config.config_dir, exist_ok=True)

    setup_seed(config.seed)
    setup_logging(config.log_dir)

    logging.info("Args: %s", config.__dict__)

    train_and_validate(config)

    config.save_dict(f"{config.config_dir}/config.json")
    
