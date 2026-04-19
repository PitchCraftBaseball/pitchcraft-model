# Model with Dropout
import torch.nn as nn
import torch

class PitchRNN(nn.Module):
    def __init__(self,
        cat_vocab_sizes:      dict,
        num_features:         int,
        emb_dims:             dict = None,
        hidden:               int  = 128,
        num_classes:          int  = 15,
        num_location_classes: int  = 3,
        num_layers:           int  = 1,
        pad_id:               int  = 0,
    ):
        super().__init__()
        self.cat_cols = list(cat_vocab_sizes.keys())

        if emb_dims is None:
            emb_dims = {col: 16 for col in self.cat_cols}

        self.embs = nn.ModuleDict({
            col: nn.Embedding(
                cat_vocab_sizes[col],
                emb_dims[col],
                padding_idx=pad_id
            )
            for col in self.cat_cols
        })

        in_dim   = sum(emb_dims.values()) + num_features
        self.rnn = nn.RNN(
            in_dim, hidden,
            batch_first=True,
            num_layers=num_layers
        )

        self.fc_pitch = nn.Linear(hidden, num_classes)
        self.fc_horiz = nn.Linear(hidden, num_location_classes)
        self.fc_vert  = nn.Linear(hidden, num_location_classes)

    def forward(self, x_cat, x_num):
        embs = [
            self.embs[col](x_cat[:, :, j])
            for j, col in enumerate(self.cat_cols)
        ]
        x    = torch.cat(embs + [x_num], dim=-1)
        h, _ = self.rnn(x)

        return self.fc_pitch(h), self.fc_horiz(h), self.fc_vert(h)