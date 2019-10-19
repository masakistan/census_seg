import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from os.path import join
from os import makedirs, walk
from torch.utils.data import DataLoader, Dataset
from histo import process


START, END = 260, 670
THRESH = 300000
VTHRESH = 25000
PAD = 0

EXPECTED_HEIGHT = 37
MERGED_ROW_TOLERANCE = 10

class FormsDataset(Dataset):
    def __init__(self, img_dir, out_dir):
        self.paths = []
        for r, d, f in walk(img_dir):
            for file in f:
                if '.jpg' in file:
                    self.paths.append((r, file))

        print('found', len(self.paths), 'images')
        self.paths.sort()
        self.out_dir = out_dir

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = join(*self.paths[idx])
        process(img_path, self.out_dir)
        return img_path

    
img_dir = sys.argv[1]
out_dir = sys.argv[2]
img_dataset = FormsDataset(img_dir, out_dir)

img_dataloader = DataLoader(
    img_dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 60,
)


total = len(img_dataloader)
for idx, path in enumerate(img_dataloader):
    print(idx, '/', total, path)

#cv2.imwrite(out_path, name)
