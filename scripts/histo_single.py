import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from os.path import join
from os import makedirs, walk
from torch.utils.data import DataLoader, Dataset
from histo import process

img_path = sys.argv[1]
out_dir = sys.argv[2]
process(img_path, out_dir)
