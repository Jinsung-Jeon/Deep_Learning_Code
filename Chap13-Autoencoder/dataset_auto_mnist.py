# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:51:34 2020

@author: Jinsung
"""

import sys
sys.path.insert(0, 'C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Chap5-Classification Flower')
from dataset import *

class AutoencoderDataset(Dataset):
    def __init__(self, name, mode, train_ratio=1.0):
        self.train_ratio = train_ratio
        super(AutoencoderDataset, self).__init__(name, mode)
        
    def get_autoencode_data(self, batch_size, nth):
        xs, ys = self.get_train_data(batch_size, nth)
        return xs
    
    @property
    def tarin_count(self):
        return int(len(self.tr_xs) * self.train_ratio)
    
    @property
    def autoencode_count(self):
        return len(self.tr_xs)
    
class MnistAutoDataset(AutoencoderDataset):
    def __init__(self, train_ratio=0.1):
        super(MnistAutoDataset, self).__init__('mnist', 'select', train_ratio)
        
        tr_x_path = 'C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Datasets\\mnist\\train-images.idx3-ubyte'
        tr_y_path = 'C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Datasets\\mnist\\train-labels.idx1-ubyte'
        
        xs = np.fromfile(tr_x_path, dtype='uint8')[16:]
        ys = np.fromfile(tr_y_path, dtype='uint8')[8:]
        
        xs = xs.reshape([-1, 28*28])
        ys = np.eye(10)[ys]
        
        self.shuffle_data(xs, ys)
        
def mnist_visualize(self, xs, estimates, answers):
    dump_text(answers, estimages)
    dump_image_data(xs)
    
def mnist_autoencode_visualize(self, xs, rep, estimates, answers):
    dump_text(answers, estimates)
    dump_image_data(xs)
    dump_image_data(rep)
    
def mnist_hash_result_visulaize(self, images):
    dump_image_data(images)
    
def dump_text(answers, estimates):
    ans = np.argmax(answers, axis=1)
    est = np.argmax(estimates, axis=1)
    print('정답', ans, ' vs. ', '추정', est)
    
def dump_image_data(images):
    show_cnt = len(images)
    fig, axes = plt.subplots(1, show_cnt, figsize=(show_cnt, 1))
    
    for n in range(show_cnt):
        plt.subplot(1, show_cnt, n+1)
        plt.imshow(images[n].reshape(28, 28), cmap='Greys_r')
        plt.axis('off')
        
    plt.draw()
    plt.show()
    
MnistAutoDataset.visualize = mnist_visualize
MnistAutoDataset.autoencode_visualize = mnist_autoencode_visualize
MnistAutoDataset.hash_result_visualize = mnist_hash_result_visulaize