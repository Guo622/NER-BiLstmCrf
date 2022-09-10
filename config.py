import json
import torch


class Config():
    def __init__(self) -> None:

        self.seed = 42
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.log_dir = './log/'
        self.save_dir = './save/'
        self.cache_dir = './cache/'
        self.config_dir = './config/'

        self.vocab_size = None  #dataset中设置

        self.weight_decay = 0.00
        self.learning_rate = 0.01
        self.min_lr = 1e-6
        self.adam_epsilon = 1e-8
        self.train_batch_size = 32
        self.val_batch_size = 128
        self.test_batch_size = 128
        self.num_layers = 2
        self.max_epochs = 5
        self.print_steps = 20

        self.embedding_size = 300
        self.hidden_size = 300
        self.dropout = 0.2
        
        self.checkpoint = None
        

    def save_dict(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=4)

    def load_from_dict(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            dic = json.load(f)
        for key, value in dic.items():
            self.__setattr__(key,value)