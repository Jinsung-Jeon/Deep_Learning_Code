# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:49:08 2020

@author: Jinsung
"""
import sys
sys.path.insert(0, 'C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Chap5- Classification Flower')
from mathutil import *

class AdamModel(MlpModel):
    def __init__(self, name, dataset, hconfigs):
        sefl.use_adam = False
        super(AdamModel, self).__init__(name, dataset, hconfigs)
        
def adam_backprop_layer(self, G_y, hconfig, pm, aux):
    x, y = aux
    if hconfig is not None:
        
        G_y = relu_derv(y) * G_y
        
        g_y_weight = x.transpose()
        g_y_input = pm['w'].transpose()
        
        G_weight = np.matmul(g_y_weight, G_y)
        G_bias = np.sum(G_y, axis=0)
        G_input = np.matmul(G_y, g_y_input)
        
        self.update_param(pm, 'w', G_weight)
        self.update_param(pm, 'b', G_bias)
        
        return G_input

AdamModel.backprop_layer = adam_backprop_layer

def adam_update_param(self, pm, key, delta):
    if self.use_adam:
        delta = self.eval_adam_delta(pm, key, delta)
        
    pm[key] -= self.learning_rate*delta

AdamModel.update_param = adam_update_param

def adam_eval_adam_delta(self, pm, key, delta):
    ro_1 = 0.9
    ro_2 = 0.999
    epsilon = 1.0e-8
    
    skey, tkey, step = 's' + key, 't' + key, 'n' + key
    if skey not in pm:
        pm[skey] = np.zeros(pm[key].shape)
        pm[tkey] = np.zeros(pm[key].shape)
        pm[step] = 0
        
    s = pm[skey] = ro_1 * pm[skey] + (1 - ro_1) * delta
    t = pm[tkey] = ro_2 * pm[tkey] + (1 - ro_2) * (delta * delta)
    
    pm[step] += 1
    
    s = s / (1 - np.power(ro_1, pm[step]))
    t = t / (1 - np.power(ro_2, pm[step]))
    
    return s / (np.sqrt(t)+epsilon)

AdamModel.eval_adam_delta = adam_eval_adam_delta