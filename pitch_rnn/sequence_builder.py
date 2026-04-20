from torch.utils.data import Dataset

# Create the Dataset
class PitchSeqDS(Dataset):
    def __init__(self, Xc, Xn, Y):
        self.Xc, self.Xn, self.Y = Xc, Xn, Y
    def __len__(self): return self.Y.size(0)
    def __getitem__(self, i): return self.Xc[i], self.Xn[i], self.Y[i]