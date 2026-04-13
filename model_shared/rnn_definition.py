# Model with Dropout
import torch.nn as nn
import torch

class PitchRNN(nn.Module):
    def __init__(self, cat_vocab_sizes, num_features, emb_dims=None, hidden=128, num_classes=16, pad_id=0, dropout=0.3, num_layers=1):
        super().__init__()
        self.cat_cols = list(cat_vocab_sizes.keys())

        if emb_dims is None:
            emb_dims = {col: 16 for col in self.cat_cols}

        self.embs = nn.ModuleDict({
            col: nn.Embedding(cat_vocab_sizes[col], emb_dims[col], padding_idx=pad_id)
            for col in self.cat_cols
        })

        self.emb_dropout = nn.Dropout(dropout)
        
        in_dim = sum(emb_dims.values()) + num_features
        self.rnn = nn.GRU(in_dim, hidden, batch_first=True, dropout=dropout if dropout > 0 else 0, num_layers=num_layers)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc  = nn.Linear(hidden, num_classes)

    def forward(self, x_cat, x_num):
        embs = []
        for j, col in enumerate(self.cat_cols):
            embs.append(self.embs[col](x_cat[:, :, j]))  

        embs = [self.emb_dropout(emb) for emb in embs]

        x = torch.cat(embs + [x_num], dim=-1)           
        h, _ = self.rnn(x)        
        h = self.fc_dropout(h)                       
        return self.fc(h)                            