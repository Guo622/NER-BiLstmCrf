import torch
import torch.nn as nn
from config import Config
from dataset import TAG2IDX, START_TAG_IDX, END_TAG_IDX
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLstmCrf(nn.Module):
    def __init__(self, config:Config) -> None:
        super().__init__()
        vocab_size = config.vocab_size
        embedding_size = config.embedding_size
        hidden_size = config.hidden_size
        self.tagset_size = len(TAG2IDX)
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=True,
            bidirectional=True
        )
        self.projection = nn.Linear(hidden_size*2, self.tagset_size)
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        nn.init.normal_(self.transitions)
        self.transitions.detach()[START_TAG_IDX, :] = -10000
        self.transitions.detach()[:, END_TAG_IDX] = -10000

    def forward_alpha(self, emission, mask):
        bs, l = mask.size()
        alpha_init = torch.full((bs, self.tagset_size), fill_value=-10000).to(emission.device)
        alpha_init[:, START_TAG_IDX] = 0

        alpha = alpha_init
        for t in range(l):
            mask_t = mask[:, t].unsqueeze(-1)
            emission_t = emission[:, t, :].unsqueeze(1).expand(-1, self.tagset_size, -1) # 行扩展
            alpha_matrix = alpha.unsqueeze(2).expand(-1, -1, self.tagset_size) #列扩展
            add_matrix = alpha_matrix + emission_t + self.transitions
            alpha = torch.logsumexp(add_matrix, dim=1)*mask_t + alpha*(1-mask_t) # mask为1则更新否则保留原值
        alpha = alpha + self.transitions[:, END_TAG_IDX].unsqueeze(0)
        return torch.logsumexp(alpha, dim=1) # [bs,]

    def score_sentence(self, emission, labels, length, mask):
        bs, l = mask.size()
        start_tag_batch = labels.new_full((bs, 1), fill_value=START_TAG_IDX)
        labels = torch.cat([start_tag_batch, labels], dim=1)
        scores = torch.zeros(bs).to(emission.device)

        for t in range(l):
            mask_t = mask[:, t]
            emission_t = emission[:, t, :]
            label_t_ = labels[:, t+1]
            emit_score = torch.cat([
                emit[next_label].unsqueeze(-1) for emit, next_label in zip(emission_t, label_t_)
            ], dim=0)
            transition_score = torch.stack([
                self.transitions[labels[b, t], labels[b, t+1]] for b in range(bs)
            ])
            scores += (emit_score+transition_score)*mask_t
        transition_to_end = torch.stack(
            [self.transitions[label[length[b]], END_TAG_IDX] for b, label in enumerate(labels)] )
        scores += transition_to_end
        return scores

    def viterbi_decode(self, emission, length, mask):
        bs, l = mask.size()
        beta_init = torch.full((bs, self.tagset_size), fill_value=-10000).to(emission.device)
        beta_init[:, START_TAG_IDX] = 0
        pointers = []

        beta = beta_init
        for t in range(l):
            mask_t = mask[:, t].unsqueeze(-1)
            emission_t = emission[: ,t, :]
            beta_matrix = beta.unsqueeze(2).expand(-1, -1, self.tagset_size) #列扩展
            m = beta_matrix + self.transitions
            max_trans, pointer = torch.max(m, dim=2)
            pointers.append(pointer)
            beta = (max_trans + emission_t)*mask_t + beta*(1-mask_t)

        pointers = torch.stack(pointers, 1)
        beta = beta + self.transitions[END_TAG_IDX]
        best_score, best_label = torch.max(beta, dim=1)
        best_path = []
        for b in range(bs):
            length_b = length[b]
            pointers_b = pointers[b, :length_b]
            best_label_b = best_label[b]
            best_path_b = [best_label_b.item()]
            for lb in range(length_b):
                idx = length_b-lb-1
                best_label_b = pointers_b[idx][best_label_b]
                best_path_b = [best_label_b.item()] + best_path_b
            best_path_b = best_path_b[1:]
            best_path.append(best_path_b)
        return best_path


    def neg_log_likelihood(self, emission, labels, length, mask):
        alpha_loss = self.forward_alpha(emission, mask)
        score_sentence = self.score_sentence(emission, labels, length, mask)
        loss = alpha_loss - score_sentence
        return loss.sum()

    def forward(self, ids, lengths):
        embeds = self.word_embedding(ids)
        embeds_packed = pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=False)
        lstm_out= pad_packed_sequence(self.lstm(embeds_packed)[0], batch_first=True)[0]
        emission = self.projection(lstm_out)
        return emission
        