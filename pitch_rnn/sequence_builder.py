from torch.utils.data import Dataset

# Create the Dataset
class PitchSeqDS(Dataset):
    def __init__(self, X_cat, X_num, Y_pitch, Y_horiz, Y_vert):
        self.X_cat   = X_cat
        self.X_num   = X_num
        self.Y_pitch = Y_pitch
        self.Y_horiz = Y_horiz
        self.Y_vert  = Y_vert

    def __len__(self):
        return self.Y_pitch.size(0)

    def __getitem__(self, i):
        return (
            self.X_cat[i],
            self.X_num[i],
            self.Y_pitch[i],
            self.Y_horiz[i],
            self.Y_vert[i],
        )