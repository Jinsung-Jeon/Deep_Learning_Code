# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:30:37 2020

@author: Jinsung
"""


import numpy as np
import csv
import time

np.random.seed(1234)

""" 고정되지 않은 난수 발생 패턴 이용
def randomize():
    np.random.seed(time.time())
"""

#하이퍼파라미터값 정의
global hidden_config

#파라미터 초기화 함수 정의
'''
기존 SLP
def init_model():
    global weight, bias, input_cnt, output_cnt
    weight = np.random.normal(RND_MEAN, RND_STD, [input_cnt, output_cnt])
    bias = np.zeros([output_cnt])
'''
def init_model():
    if hidden_config is not None:
        print('은닉 계층 {}개를 갖는 다층 퍼셉트론이 작동되었습니다.'.format(len(hidden_config)))
        init_model_hiddens()
    else:
        print('은닉 계층 하나를를 갖는 다층 퍼셉트론이 작동되었습니다.')
        init_model_hidden1()

def init_model_hidden1():
    global pm_output, pm_hidden, input_cnt, output_cnt, hidden_cnt
    
    pm_hidden = alloc_param_pair([input_cnt, hidden_cnt])
    pm_output = alloc_param_pair([hidden_cnt, output_cnt])

def init_model_hiddens():
    global pm_output, pm_hiddens, input_cnt, output_cnt, hidden_config
    
    pm_hiddens = []
    prev_cnt = input_cnt
    
    for hidden_cnt in hidden_config:
        pm_hiddens.append(alloc_param_pair([prev_cnt, hidden_cnt]))
        prev_cnt = hidden_cnt
    
    pm_output = alloc_param_pair([prev_cnt, output_cnt])
    
def alloc_param_pair(shape):
    weight = np.random.normal(RND_MEAN, RND_STD, shape)
    bias = np.zeros(shape[-1])
    return {'w':weight, 'b':bias}

#단층 퍼셉트론에 대한 순전파 및 역전파 함수 정의
'''
기존 SLP
def forward_neuralnet(x):
    global weight, bias
    output = np.matmul(x, weight) + bias
    return output, x

def backprop_neuralnet(G_output, x):
    global weight, bias
    g_output_w = x.transpose()
    
    G_w = np.matmul(g_output_w, G_output)
    G_b = np.sum(G_output, axis=0)

    weight -= LEARNING_RATE * G_w
    bias -= LEARNING_RATE * G_b
'''
def forward_neuralnet(x):
    if hidden_config is not None:
        return forward_neuralnet_hiddens(x)
    else:
        return forward_neuralnet_hidden1(x)

def forward_neuralnet_hidden1(x):
    global pm_output, pm_hidden
    
    hidden = relu(np.matmul(x, pm_hidden['w']) + pm_hidden['b'])
    output = np.matmul(hidden, pm_output['w']) + pm_output['b']
    
    return output, [x, hidden]

def forward_neuralnet_hiddens(x):
    global pm_output, pm_hiddens
    
    hidden = x
    hiddens = [x]
    
    for pm_hidden in pm_hiddens:
        hidden = relu(np.matmul(hidden, pm_hidden['w']) + pm_hidden['b'])
        hiddens.append(hidden)
        
    output = np.matmul(hidden, pm_output['w']) + pm_output['b']
    
    return output, hiddens

def backprop_neuralnet(G_output, hiddens):
    if hidden_config is not None:
        backprop_neuralnet_hiddens(G_output, hiddens)
    else:
        backprop_neuralnet_hidden1(G_output, hiddens)
        
def backprop_neuralnet_hidden1(G_output, aux):
    global pm_output, pm_hidden
    
    x, hidden = aux
    g_output_w_out = hidden.transpose()
    G_w_out = np.matmul(g_output_w_out, G_output)
    G_b_out = np.sum(G_output, axis=0)
    
    g_output_hidden = pm_output['w'].transpose()
    G_hidden = np.matmul(G_output, g_output_hidden)
    
    pm_output['w'] -= LEARNING_RATE * G_w_out
    pm_output['b'] -= LEARNING_RATE * G_b_out
    
    G_hidden = G_hidden * relu_derv(hidden)
    
    g_hidden_w_hid = x.transpose()
    G_w_hid = np.matmul(g_hidden_w_hid, G_hidden)
    G_b_hid = np.sum(G_hidden, axis=0)
    
    pm_hidden['w'] -= LEARNING_RATE * G_w_hid
    pm_hidden['b'] -= LEARNING_RATE * G_b_hid

def backprop_neuralnet_hiddens(G_output, aux):
    global pm_output, pm_hiddens
    
    hiddens = aux
    g_output_w_out = hiddens[-1].transpose()
    G_w_out = np.matmul(g_output_w_out, G_output)
    G_b_out = np.sum(G_output, axis=0)
    
    g_output_hidden = pm_output['w'].transpose()
    G_hidden = np.matmul(G_output, g_output_hidden)
    
    pm_output['w'] -= LEARNING_RATE * G_w_out
    pm_output['b'] -= LEARNING_RATE * G_b_out
    
    for n in reversed(range(len(pm_hiddens))):
        G_hidden = G_hidden * relu_derv(hiddens[n+1])
    
        g_hidden_w_hid = hiddens[n].transpose()
        G_w_hid = np.matmul(g_hidden_w_hid, G_hidden)
        G_b_hid = np.sum(G_hidden, axis=0)
    
        g_hidden_hidden = pm_hiddens[n]['w'].transpose()
        G_hidden = np.matmul(G_hidden, g_hidden_hidden)
        
        pm_hiddens[n]['w'] -= LEARNING_RATE * G_w_hid
        pm_hiddens[n]['b'] -= LEARNING_RATE * G_b_hid
        
def relu(x):
    return np.maximum(x,0)

def relu_derv(y):
    return np.sign(y)

#은닉계층구조 지정 함수 정의
def set_hidden(info):
    global hidden_cnt, hidden_config
    if isinstance(info, int):
        hidden_cnt = info
        hidden_config = None
    else:
        hidden_config = info