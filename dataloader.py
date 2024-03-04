import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch.utils.data.sampler import Sampler

sys.path.append("..")
global kernel_sizes

def read_txt(path):
    data=[]
    present = 0
    with open(path) as f:
        for line in f.readlines():
            line= line.strip("\t").strip("\n").split("\t")
            data.append(line)
            # print(line[2:])
            if(int(line[2])==0 and int(line[3])==0 and int(line[4])==0 and int(line[5])==0):
                present = present+1

    print("data analysis", present/(len(data))*1.0)

    return data

class BaseFeeder(data.Dataset):
    def __init__(self,  mode="train", input_size=5,data_path='./data.txt'):
        self.mode = mode
        self.input_size = input_size
        self.data = read_txt(data_path)
        self.split_data(self.mode)

    def split_data(self, mode):
        data_len = len(self.data)
        if mode=="train":
            self.data_len = int(data_len * 0.8)
            self.data = self.data[:self.data_len]
        elif mode=="test":
            self.data_len=int(data_len*0.1)
            self.data = self.data[int(data_len*0.8):int(data_len*0.9)]
        else:
            self.data = self.data[int(data_len*0,9):]
            self.data_len = len(self.data)

    def str2float(self, data):
        return [float(da) for da in data]

    def __getitem__(self, idx):
        data = self.data[idx:idx+self.input_size]
        data = np.array([self.str2float(da[1:]) for da in data])
        datax =torch.tensor([[da] for da in data[:,0]])
        datay = data[:,1:]

        return datax, datay

    def __len__(self):
        return self.data_len - self.input_size


if __name__ == "__main__":
    feeder = BaseFeeder(mode='train')
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    for data in dataloader:
        x = data[0]
        print(x.shape)
        y = data[1]
        print(y.shape)
        y = y.int()
        print(y)
        break
