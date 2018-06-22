""" Module for the data loading pipeline for the model to train """

import cv2
import torch as th

from torch.utils.data import Dataset


class Face2TextDataset(Dataset):
    """ PyTorch Dataset wrapper around the Face2Text dataset """

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass