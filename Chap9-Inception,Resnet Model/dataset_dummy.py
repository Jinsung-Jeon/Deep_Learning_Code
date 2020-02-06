# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:44:04 2020

@author: Jinsung
"""
import sys
sys.path.insert(0, 'C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Chap5-Classification Flower')
from dataset import *

class DummyDataset(Dataset):
    def __init__(self, name, mode, input_shape, output_shape):
        super(DummyDataset, self).__init__(name, mode)
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.tr_xs, self.tr_ys = [], []
        self.te_xs, self.te_ys = [], []