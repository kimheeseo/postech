from PIL import Image
import glob
import os

from torch.utils.data import Dataset


class TrainSet(Dataset):
    def __init__(self, paths, lab, preproc, lab2idx, indices=None):
        self.img_paths = paths
        self.labs = lab
        self.lab2idx = lab2idx
        self.preproc = preproc
        self.indices = list(range(len(self.labs))) if indices is None else indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        lab = self.labs[idx]
        
        img = self.preproc(img)
        if img.shape[0] != 3:
            img = img[:3]
        lab = self.lab2idx[lab]
        
        return img, lab