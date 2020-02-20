# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:51:05 2020

@author: Jinsung
"""

import sys
sys.path.insert(0, 'C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Chap12-CNN+RNN')
from rnn_ext_model import *

class Autoencoder(RnnExtModel):
    def __init__(self, name, dataset, hconfigs, show_maps=False, l2_decay=0, l1_decay=0, dump_structure=False, fix_encoder=False):
        self.fix_encoder = fix_encoder
        super(Autoencoder, self).__init__(name, dataset, hconfigs, show_maps, l2_decay, l1_decay, dump_structure)
        
def autoencoder_init_parameters(self, hconfigs):
    econf = hconfigs['encoder']
    dconf = hconfigs['decoder']
    hconf = hconfigs['supervised']
    
    in_shape = self.dataset.input_shape
    
    pme, code_shape = self.build_subnet(econf, in_shape)
    pmd, represent_shape = self.build_subnet(dconf, code_shape)
    pmh, hidden_shape = self.build_subnet(hconf, code_shape)
    
    self.econfigs, self.dconfigs, self.hconfigs = econf, dconf, hconf
    self.pm_encoder, self.pm_decoder, self.pm_hiddens = pme, pmd, pmh
    
    output_cnt = int(np.prod(self.dataset.output_shape))
    self.seqout = False
    
    if len(hconf) > 0 and get_layer_type(hconf[-1]) in ['rnn', 'lstm']:
        if get_conf_param(hconf[-1], 'outseq', True):
            self.seqout = True
            hidden_shape = hidden_shape[1:]
            output_cnt = int(np.prod(self.dataset.output_shape[1:]))
            
    self.pm_output, _ = self.alloc_layer_param(hidden_shape, output_cnt)
    
def autoencoder_build_subnet(self, hconfigs, prev_shape):
    pms = []
    
    for hconfig in hconfigs:
        pm, prev_shape = self.alloc_layer_param(prev_shape, hconfig)
        pms.append(pm)
        
    return pms, prev_shape

Autoencoder.build_subnet = autoencoder_build_subnet
Autoencoder.init_parameters = autoencoder_init_parameters

# 오토인코딩 단계의 메인 학습 메서드 정의
def autoencoder_autoencode(self, epoch_count=10, batch_size=10, learning_rate=0.001, report=0):
    self.learning_rate = learning_rate
    
    batch_count = self.dataset.autoencode_count // batch_size
    time1 = time2 = int(time.time())
    
    if report != 0:
        print("Model {} autoencode started:".format(self.name))
        
    for epoch in range(epoch_count):
        costs = []
        accs = []
        self.dataset.shuffle_train_data(batch_size*batch_count)
        for n in range(batch_count):
            trX = self.dataset.get_autoencode_data(batch_size, n)
            cost, acc = self.autoencode_step(trX)
            costs.append(cost)
            accs.append(acc)
            
        if report > 0 and (epoch+1) % report == 0:
            acc = np.mean(accs)
            time3 = int(time.time())
            tm1, tm2 = time3-time2, time3-time1
            
            self.dataset.train_prt_result(epoch+1, costs, accs, acc, tm1, tm2)
            time2 = time3
            
    tm_total = int(time.time()) - time1
    if report != 0:
        print("Model {} autoencode ended in {} secs:".format(self.name, tm_total))
        
Autoencoder.autoencode = autoencoder_autoencode

# 미니배치 단위의 오토인코딩 학습 메서드 정의
def autoencoder_autoencode_step(self, x):
    self.is_training = True
    
    hidden, aux_encoder, aux_decoder = self.forward_autoencode(x)
    
    diff = hidden - x
    square = np.square(diff)
    loss = np.mean(square)
    
    mse = np.mean(np.square(hidden - x))
    accuracy = 1 - np.sqrt(mse) / np.mean(x)
    
    g_loss_square = np.ones(x.shape) / np.prod(x.shape)
    g_square_diff = 2*diff
    g_diff_output = 1
    
    G_loss = 1.0
    G_square = g_loss_square * G_loss
    G_diff = g_square_diff * G_square
    G_hidden = g_diff_output * G_diff
    
    self.backprop_autoencode(G_hidden, aux_encoder, aux_decoder)
    
    self.is_training = False
    
    return loss, accuracy

Autoencoder.autoencode_step = autoencoder_autoencode_step

# 오토인코딩 순전파 처리 메서드 정의
def autoencoder_forward_autoencode(self, x):
    hidden = x
    aux_encoder, aux_decoder = [], []
    
    for n, hconfig in enumerate(self.econfigs):
        hidden, aux = self.forward_layer(hidden, hconfig, self.pm_encoder[n])
        aux_encoder.append(aux)
        
    for n, hconfig in enumerate(self.dconfigs):
        hidden, aux = self.forward_layer(hidden, hconfig, self.pm_decoder[n])
        aux_decoder.append(aux)
        
    return hidden, aux_encoder, aux_decoder

Autoencoder.forward_autoencode = autoencoder_forward_autoencode

def autoencoder_backprop_autoencode(self, G_hidden, aux_encoder, aux_decoder):
    for n in reversed(range(len(self.dconfigs))):
        hconfig, pm, aux = self.dconfigs[n], self.pm_decoder[n], aux_decoder[n]
        G_hidden = self.backprop_layer(G_hidden, hconfig, pm, aux)
        
    for n in reversed(range(len(self.econfigs))):
        hconfig, pm, aux = self.econfigs[n], self.pm_encoder[n], aux_encoder[n]
        G_hidden = self.backprop_layer(G_hidden, hconfig, pm, aux)
        
Autoencoder.backprop_autoencode = autoencoder_backprop_autoencode

# 지도학습을 위한 순전파 처리 메서드 재정의
def autoencoder_forward_neuralnet(self, x):
    hidden = x
    
    aux_encoder = []
    
    for n, hconfig in enumerate(self.econfigs):
        hidden, aux = self.forward_layer(hidden, hconfig, self.pm_encoder[n])
        aux_encoder.append(aux)
        
    output, aux_layers = super(Autoencoder, self).forward_neuralnet(hidden)

    return output, [aux_encoder, aux_layers]

Autoencoder.forward_neuralnet = autoencoder_forward_neuralnet

# 지도학습을 위한 역전파 처리 메서드 재정의
def autoencoder_backprop_neuralnet(self, G_output, aux):
    aux_encoder, aux_layers = aux
    
    G_hidden = super(Autoencoder, self).backprop_neuralnet(G_output, aux_layers)
    
    if self.fix_encoder:
        return G_hidden
    
    for n in reversed(range(len(self.econfigs))):
        hconfig, pm, aux = self.econfigs[n], self.pm_encoder[n], aux_encoder[n]
        G_hidden = self.backprop_layer(G_hidden, hconfig, pm, aux)
        
    return G_hidden

Autoencoder.backprop_neuralnet = autoencoder_backprop_neuralnet

# 시맨틱 해싱을 위한 인덱싱 메서드 정의
def autoencoder_semantic_hasing_index(self):
    self.hash_data = self.dataset.tr_xs
    self.hash_table = {}
    
    hidden = self.hash_data
    for n, hconfig in enumerate(self.econfigs):
        hidden, _ = self.forward_layer(hidden, hconfig, self.pm_encoder[n])
        
    self.bit_weight = [np.power(2,n) for n in range(hidden.shape[-1])]
    
    for n, code in enumerate(hidden):
        bin_code = np.around(code)
        hash_idx = int(np.sum(self.bit_weight * bin_code))
        if hash_idx not in self.hash_table:
            self.hash_table[hash_idx] = []
        self.hash_table[hash_idx].append([n, code])
        
Autoencoder.semantic_hashing_index = autoencoder_semantic_hasing_index

# 시맨틱 해싱을 위한 검색 메서드 정의
def autoencoder_semantic_hashing_search(self, show_cnt=3, max_rank=5):
    data_cnt, data_size = self.hash_data.shape
    nths = np.random.randint(data_cnt, size=show_cnt)
    hidden = self.hash_data[nths]
    for n, hconfig in enumerate(self.econfigs):
        hidden, _ = self.forward_layer(hidden, hconfig, self.pm_encoder[n])
    bin_codes = np.around(hidden)
    hash_idxs = np.sum(bin_codes * self.bit_weight, axis=1)
    
    for n in range(show_cnt):
        fetched = self.hash_table[int(hash_idxs[n])]
        codes = [lst[1] for lst in fetched]
        diff = np.sum(np.square(codes-hidden[n]), axis=1)
        merged = [[lst[0], diff[m]] for m, lst in enumerate(fetched)]
        merged.sort(key = take_diff)
        
        buffer = np.zeros([max_rank+1, data_size])
        buffer[0] = self.hash_data[nths[n]]
        for k, pair in enumerate(merged):
            if k >= max_rank:
                break
            buffer[k+1] = self.bash_data[pair[0]]
            self.dataset.hash_result_visualize(buffer)
            
def take_diff(element):
    return element[1]

Autoencoder.semantic_hashing_search = autoencoder_semantic_hashing_search

def autoencoder_visualize(self, num):
    print("Model {} Visualization".format(self.name))
    deX, deY = self.dataset.get_visualize_data(num)
    copy, _, _ = self.forward_autoencode(deX)
    est = self.get_estimate(deX)
    self.dataset.autoencode_visualize(deX, copy, est, deY)
    
Autoencoder.visualize = autoencoder_visualize