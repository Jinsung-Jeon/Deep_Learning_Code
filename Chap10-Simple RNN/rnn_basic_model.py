# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:06:32 2020

@author: Jinsung
"""

import sys
sys.path.insert(0, 'C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Chap9-Inception,Resnet Model')
from cnn_ext_model import *

class RnnBasicModel(CnnExtModel):
    pass

# 순환 계층에 대한 파라미터 생성 메서드 정의하기
def rnn_basic_alloc_rnn_layer(self, input_shape, hconfig):
    inseq = get_conf_param(hconfig, 'inseq', True)
    outseq = get_conf_param(hconfig, 'outseq', True)
    
    if inseq:
        timesteps1, timefeats = input_shape
    else:
        timesteps1 = get_conf_param(hconfig, 'timesteps') + 1
        timefeats = np.prod(input_shape)
    
    recur_size = get_conf_param(hconfig, 'recur_size')
    
    ex_inp_dim = timefeats + recur_size
    weight, bias = self.alloc_param_pair([ex_inp_dim, recur_size])
    
    if outseq:
        output_shape = [timesteps1, recur_size]
    else:
        output_shape = [recur_size]
        
    rnn_info = [inseq, outseq, timesteps1, timefeats, recur_size]
    
    return {'w':weight, 'b':bias, 'info':rnn_info}, output_shape

RnnBasicModel.alloc_rnn_layer = rnn_basic_alloc_rnn_layer

# 순환 계층에 대한 파라미터 순전파 메서드 정의하기
def rnn_basic_forward_rnn_layer(self, x, hconfig, pm):
    inseq, outseq, timesteps1, timefeats, recur_size = pm['info']
    mb_size = x.shape[0]
    
    if inseq:
        x_slices = x[:, 1:, :].transpose([1,0,2])
        lengths = x[:, 0, 0].astype(np.int32)
        timesteps = np.max(lengths)
    else:
        x_slice = x
        timesteps = timesteps1 - 1
        lengths = [timesteps] * mb_size
        
    recurrent = np.zeros([mb_size, recur_size])
    outputs, aux_steps = [], []
    
    for n in range(timesteps):
        if inseq :
            x_slice = x_slices[n]
        ex_inp = np.hstack([x_slice, recurrent])
        affine = np.matmul(ex_inp, pm['w']) + pm['b']
        recurrent = self.activate(affine, hconfig)
        
        outputs.append(recurrent)
        aux_steps.append(ex_inp)
        
    if outseq:
        output = np.zeros([mb_size, timesteps1, recur_size])
        output[:, 0, 0] = lengths
        output[:, 1:, :] = np.asarray(outputs).transpose([1, 0, 2])
    else:
        output = np.zeros([mb_size, recur_size])
        for n in range(mb_size):
            output[n] = outputs[lengths[n]-1][n]
            
    return output, [x, lengths, timesteps, outputs, aux_steps]

RnnBasicModel.forward_rnn_layer = rnn_basic_forward_rnn_layer

# 순환 계층에 대한 파라미터 역전파 메서드 정의하기
def rnn_basic_backprop_rnn_layer(self, G_y, hconfig, pm, aux):
    inseq, outseq, timesteps1, timefeats, recur_size = pm['info']
    x, lengths, timesteps, outputs, aux_steps = aux
    mb_size = x.shape[0]
    
    G_weight = np.zeros_like(pm['w'])
    G_bias = np.zeros_like(pm['b'])
    G_x = np.zeros(x.shape)
    G_recurrent = np.zeros([mb_size, recur_size])
    
    if inseq:
        G_x[:, 0, 0] = lengths
        
    if outseq:
        G_outputs = G_y[:, 1:, :].transpose([1, 0, 2])
    else:
        G_outputs = np.zeros([timesteps, mb_size, recur_size])
        for n in range(mb_size):
            G_outputs[lengths[n]-1, n, :] = G_y[n]
    
    for n in reversed(range(0, timesteps)):
        G_recurrent += G_outputs[n]
        
        ex_inp = aux_steps[n]
        
        G_affine = self.activate_derv(G_recurrent, outputs[n], hconfig)
        
        g_affine_weight = ex_inp.transpose()
        g_affine_input = pm['w'].transpose()
        
        G_weight += np.matmul(g_affine_weight, G_affine)
        G_bias += np.sum(G_affine, axis = 0)
        G_ex_inp = np.matmul(G_affine, g_affine_input)
        
        if inseq:
            G_x[:,n+1,:] = G_ex_inp[:, :timefeats]
        else:
            G_x[:,:] += G_ex_inp[:, :timefeats]
            
        G_recurrent = G_ex_inp[:, timefeats:]
        
    self.update_param(pm, 'w', G_weight)
    self.update_param(pm, 'b', G_bias)
    
    return G_x

RnnBasicModel.backprop_rnn_layer = rnn_basic_backprop_rnn_layer