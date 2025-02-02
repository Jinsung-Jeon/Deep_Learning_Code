# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:12:34 2020

@author: Jinsung
"""
from dataset import *
from mathutil import *
import numpy as np

class FlowersDataset(Dataset):
    pass

def flowers_init(self, resolution=[100,100], input_shape =[-1]):
    super(FlowersDataset, self).__init__('flowers', 'select')
    
    path = 'C:/Users/Jinsung/Documents/Deep_Learning_Code/Datasets/flowers/flowers'
    self.target_names = list_dir(path)
    
    images = []
    idxs = []
    
    for dx, dname in enumerate(self.target_names):
        subpath = path + '/' + dname
        filenames = list_dir(subpath)
        for fname in filenames:
            if fname[-4:] != '.jpg':
                continue
            imagepath = os.path.join(subpath, fname)
            pixels = load_image_pixels(imagepath, resolution, input_shape)
            images.append(pixels)
            idxs.append(dx)
            
    self.image_shape = resolution + [3]
    
    xs = np.asarray(images, np.float32)
    ys = onehot(idxs, len(self.target_names))
    
    self.shuffle_data(xs, ys, 0.8)
    
FlowersDataset.__init__ = flowers_init

def flowers_visualize(self, xs, estimates, answers):
    draw_images_horz(xs, self.image_shape)
    show_select_results(estimates, answers, self.target_names)
    
FlowersDataset.visualize = flowers_visualize
