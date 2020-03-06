# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:19:48 2020

@author: Jinsung
"""

import sys
sys.path.insert(0, 'C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Chap5-Classification Flower')
from dataset import *

class GanDataset(Dataset):
    pass

# GanDataset 클래스의 시각화 메서드 정의
def gan_visualize(self, xs):
    show_cnt = len(xs)
    fig, axes = plt.subplots(1, show_cnt, figsize=(show_cnt, 1))
    
    for n in range(show_cnt):
        plt.subplot(1, show_cnt, n+1)
        if xs[n].shape[0] == 28*28:
            plt.imshow(xs[n].reshape(28,28), cmap='Greys_r')
        else:
            plt.imshow(xs[n].reshape([32,32,3]))
        plt.axis('off')
        
    plt.draw()
    plt.show()
    
GanDataset.visualize = gan_visualize

# GanDataset 클래스의 출력 메서드 재정의
def gan_train_prt_result(self, epoch, costs, accs, acc, time1, time2):
    dcost, gcost = np.mean(costs, axis=0)
    dacc, gacc = acc
    print(' Epoch {} : cost={:5.3f}/{:5.3f} acc={:5.3f}+{:5.3f}({}/{} secs)'.format(epoch, dcost, gcost, dacc, gacc, time1, time2))

def gan_test_prt_result(self, name, acc, time):
    dacc, gacc = acc
    print(' Model {} test report :accuracy = {:5.3f}/{:5.3f}, ({}secs)'.format(name, dacc, gacc, time))
    
GanDataset.train_prt_result = gan_train_prt_result
GanDataset.test_prt_result = gan_test_prt_result

# GanDatasetPicture 클래스 선언
class GanDatasetPicture(Dataset):
    def __init__(self, fname):
        super(GanDatasetPicture, self).__init__('pic_'+fname, 'binary')
        pic_path = 'C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Datasets\\pictures\\'+fname
        jpgfile = Image.open(pic_path)
        pixels = np.array(jpgfile)
        
        hn = pixels.shape[0] // 32
        wn = pixels.shape[1] // 32

        pieces = pixels[0:hn*32, 0:wn*32, 0:3]
        pieces = pieces.reshape([hn,32,wn,32,3])
        pieces = pieces.transpose([0,2,1,3,4])
        pieces = pieces.reshape([-1,32*32*3])
        
        pieces = pieces / 255.0
        
        self.shuffle_data(pieces, pieces)
        
# GanDatasetMnist 클래스 선언
class GandatasetMnist(Dataset):
    def __init__(self, name, max_cnt=0, nums=None):
        super(GanDatasetMnist, self).__init__(name, 'binary')
        
        tr_x_path = 'C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Datasets\\mnist\\train-images.idx3-ubyte'
        tr_y_path = 'C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Datasets\\mnist\\train-labels.idx1-ubyte'
        
        images = np.fromfile(tr_x_path, dtype='uint8')[16:]
        labels = np.fromfile(tr_y_path, dtype='uint8')[8:]
            
        images = images.reshape([-1, 28*28])
        images = (images - 127.5) / 127.5
        
        if max_cnt == 0:
            max_cnt = len(images)
            
        if nums in None:
            xs = images[:max_cnt]
        else:
            ids = []
            for n in range(len(images)):
                if labels[n] in nums:
                    ids.append(n)
                if len(ids) >= max_cnt :
                    break
            xs = images[ids]
            
        self.shuffle_data(xs, xs)
        
        return images.reshape([-1, 28*28]), labels
          