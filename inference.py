import torch
from config import Config
from model import BiLstmCrf
from train import validate
from dataset import build_dataloaders


def inference(config: Config, checkpoint: str):
    _,_,test_loader,_ = build_dataloaders(config)
    model = BiLstmCrf(config)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.to(config.device)
    f1_micro = validate(model, test_loader)
    print(f"Test f1_score: {f1_micro:.3f}")


if __name__ == '__main__':
    config = Config()
    inference(config, './save/epoch4_0.821.pth')
