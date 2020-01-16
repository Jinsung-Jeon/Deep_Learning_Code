# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 20:25:42 2020

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
RND_MEAN = 0
RND_STD = 0.0030

LEARNING_RATE = 0.001

#실험용 메인 함수 정의
def abalone_exec(epoch_count=10, mb_size=10, report=1):
    load_abalone_dataset()
    init_model() #파라미터 초기화
    train_and_test(epoch_count, mb_size, report)
    
#데이터 적재 함수 정의
def load_abalone_dataset():
    with open('C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Datasets\\abalone-dataset\\abalone.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None) #첫 행은 읽지 않고 건너뛴다
        rows = []
        for row in csvreader:
            rows.append(row)
        
        global data, input_cnt, output_cnt
        input_cnt, output_cnt = 10, 1
        data = np.zeros([len(rows), input_cnt+output_cnt])
        
        for n, row in enumerate(rows):
            if row[0] == 'I':
                data[n, 0] = 1
            if row[0] == 'M':
                data[n, 1] = 1
            if row[0] == 'F':
                data[n, 2] = 1
            data[n, 3:] = row[1:]
            
#파라미터 초기화 함수 정의
def init_model():
    global weight, bias, input_cnt, output_cnt
    weight = np.random.normal(RND_MEAN, RND_STD, [input_cnt, output_cnt])
    bias = np.zeros([output_cnt])
    
#학습 및 평가 함수 정의
def train_and_test(epoch_count, mb_size, report):
    step_count = arrange_data(mb_size)
    test_x, test_y = get_test_data()
    
    for epoch in range(epoch_count):
        losses, accs = [], []
        
        for n in range(step_count):
            train_x, train_y = get_train_data(mb_size, n)
            loss, acc = run_train(train_x, train_y)
            losses.append(loss)
            accs.append(acc)
            
        if report > 0 and (epoch+1) % report == 0:
            acc = run_test(test_x, test_y)
            print('Epoch {}: loss ={:5.3f}, accuracy ={:5.3f}/{:5.3f}'.format(epoch+1, np.mean(losses), np.mean(accs),acc))
        
    final_acc = run_test(test_x, test_y)
    print('\nFinal Test: final accuracy = {:5.3f}'.format(final_acc))
    
#학습 및 평가 데이터 획득 함수 정의
def arrange_data(mb_size):
    global data, shuffle_map, test_begin_idx
    shuffle_map = np.arange(data.shape[0])
    np.random.shuffle(shuffle_map)
    step_count = int(data.shape[0]*0.8) // mb_size
    test_begin_idx = step_count * mb_size
    return step_count

def get_test_data():
    global data, shuffle_map, test_begin_idx, output_cnt
    test_data = data[shuffle_map[test_begin_idx:]]
    return test_data[:, :-output_cnt], test_data[:, -output_cnt:]

def get_train_data(mb_size, nth):
    global data, shuffle_map, test_begin_idx, output_cnt
    if nth == 0:
        np.random.shuffle(shuffle_map[:test_begin_idx])
    train_data = data[shuffle_map[mb_size*nth:mb_size*(nth+1)]]
    return train_data[:, :-output_cnt], train_data[:, -output_cnt:]

#학습 실행 함수와 평가 실행 함수 정의
def run_train(x, y):
    output, aux_nn = forward_neuralnet(x)
    loss, aux_pp = forward_postproc(output, y)
    accuracy = eval_accuracy(output, y)
    
    G_loss = 1.0
    G_output = backprop_postproc(G_loss, aux_pp)
    backprop_neuralnet(G_output, aux_nn)
    
    return loss, accuracy

def run_test(x, y):
    output, _ = forward_neuralnet(x)
    accuracy = eval_accuracy(output, y)
    return accuracy

#단층 퍼셉트론에 대한 순전파 및 역전파 함수 정의
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
    
#후처리 과정에 대한 순전파 및 역전파 함수 정의
def forward_postproc(output, y):
    diff = output - y
    square = np.square(diff)
    loss = np.mean(square)
    return loss, diff

def backprop_postproc(G_loss, diff):
    shape = diff.shape
    
    g_loss_square = np.ones(shape) / np.prod(shape)
    g_square_diff = 2 * diff
    g_diff_output = 1

    G_square = g_loss_square * G_loss
    G_diff = g_square_diff * G_square
    G_output = g_diff_output * G_diff
    
    return G_output

"""
def backprop_postproc_oneline(G_loss, diff):
    return 2*diff / np.prod(diff.shape)
"""

#정확도 계산 함수 정의
def eval_accuracy(output, y):
    mdiff = np.mean(np.abs((output - y)/y))
    return 1 - mdiff

abalone_exec()