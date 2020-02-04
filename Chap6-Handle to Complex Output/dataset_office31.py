# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:23:52 2020

@author: Jinsung
"""

import sys
sys.path.insert(0, 'C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Chap5-Classification Flower')
from dataset import *

class Office31Dataset(Dataset):
    @property
    def base(self):
        return super(Office31Dataset, self)
    
def office31_init(self, resolution=[100, 100], input_shape=[-1]):
    self.base.__init__('office31', 'dual_select')
    
    path = 'C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Datasets\\office31\\domain_adaptation_images'
    domain_names = list_dir(path)
    
    images = []
    didxs, oidxs = [], []
    
    for dx, dname in enumerate(domain_names):
        domainpath = os.path.join(path, dname, 'images')
        object_names = list_dir(domainpath)
        for ox, oname in enumerate(object_names):
            objectpath = os.path.join(domainpath, oname)
            filenames = list_dir(objectpath)
            for fname in filenames:
                if fname[-4:] != '.jpg':
                    continue
                imagepath = os.path.join(objectpath, fname)
                pixels = load_image_pixels(imagepath, resolution, input_shape)
                images.append(pixels)
                didxs.append(dx)
                oidxs.append(ox)
                
    self.image_shape = resolution + [3]
    xs = np.asarray(images, np.float32)
    
    ys0 = onehot(didxs, len(domain_names))
    ys1 = onehot(oidxs, len(object_names))
    ys = np.hstack([ys0, ys1])
    
    self.shuffle_data(xs, ys, 0.8)
    self.target_names = [domain_names, object_names]
    self.cnts = [len(domain_names)]
    
def office31_forward_postproc(self, output, y):
    outputs, ys = np.hsplit(output, self.cnts), np.hsplit(y, self.cnts)
    
    loss0, aux0 = self.base.forward_postproc(outputs[0], ys[0], 'select')
    loss1, aux1 = self.base.forward_postproc(outputs[1], ys[1], 'select')
    
    return loss0 + loss1, [aux0, aux1]

def office31_backprop_postproc(self, G_loss, aux):
    aux0, aux1 = aux
    
    G_output0 = self.base.backprop_postproc(G_loss, aux0, 'select')
    G_output1 = self.base.backprop_postproc(G_loss, aux1, 'select')
    
    return np.hstack([G_output0, G_output1])

def office31_eval_accuracy(self, x, y, output):
    outputs, ys = np.hsplit(output, self.cnts), np.hsplit(y, self.cnts)
    
    acc0 = self.base.eval_accuracy(x, ys[0], outputs[0], 'select')
    acc1 = self.base.eval_accuracy(x, ys[1], outputs[1], 'select')
    
    return [acc0, acc1]

def office31_train_prt_result(self, epoch, costs, accs, acc, time1, time2):
    acc_pair = np.mean(accs, axis=0)
    print('Epoch {}: cost={:5.3f}, accuracy={:5.3f}+{:5.3f}/{:5.3f}+{:5.3f} ({}/{} secs)'.format(epoch, np.mean(costs), acc_pair[0], acc_pair[1], acc[0], acc[1], time1, time2))
    
def adam_test_prt_result(self, name, acc, time):
    print('Model {} test report: accuracy = {:5.3f}+{:5.3f}, ({} secs)\n'.format(name, acc[0], acc[1], time))
    
def office31_get_estimate(self, output):
    outputs = np.hsplit(output, self.cnts)
    
    estimate0 = self.base.get_estimate(outputs[0], 'select')
    estimate1 = self.base.get_estimate(outputs[1], 'select')
    
    return np.hstack([estimate0, estimate1])

def office31_visualize(self, xs, estimates, answers):
    draw_images_horz(xs, self.image_shape)
    
    ests, anss = np.hsplit(estimates, self.cnts), np.hsplit(answers, self.cnts)
    
    captions = ['도메인', '상품']
    
    for m in range(2):
        print('[ {} 추정결과 ]'.format(captions[m]))
        show_select_results(ests[m], anss[m], self.target_names[m], 8)
        
Office31Dataset.__init__ = office31_init
Office31Dataset.forward_postproc = office31_forward_postproc
Office31Dataset.backprop_postproc = office31_backprop_postproc
Office31Dataset.eval_accuracy = office31_eval_accuracy
Office31Dataset.get_estimate = office31_get_estimate
Office31Dataset.train_prt_result = office31_train_prt_result
Office31Dataset.test_prt_result = adam_test_prt_result
Office31Dataset.visualize = office31_visualize