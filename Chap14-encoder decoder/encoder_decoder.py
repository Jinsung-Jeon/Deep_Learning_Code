# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:06:27 2020

@author: Jinsung
"""

import sys
sys.path.insert(0, 'C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Chap13-Autoencoder')
from autoencoder import *

class EncoderDecoder(RnnExtModel):
    pass

# 파라미터 생성 메서드 재정의
def endec_init_parameters(self, hconfigs):
    econf = hconfigs['encoder']
    dconf = hconfigs['decoder']
    
    in_shape = self.dataset.input_shape
    
    pme, code_shape = self.build_subnet(econf, in_shape)
    pmd, hidden_shape = self.build_subnet(dconf, code_shape)
    
    self.econfigs, self.dconfigs = econf, dconf
    self.pm_encoder, self.pm_decoder = pme, pmd
    
EncoderDecoder.build_subnet = autoencoder_build_subnet
EncoderDecoder.init_parameters = endec_init_parameters

# 학습 모드 제어를 위한 여러 가지 메서드 정의
def endec_set_train_mode(self, train_mode):
    self.train_mode = train_mode
    self.dataset.set_train_mode(train_mode)
    
def endec_step(self, epoch_count=10, batch_size=10, learning_rate=0.001, report=0, show_cnt=3, train_mode='both'):
    self.set_train_mode(train_mode)
    self.train(epoch_count, batch_size, learning_rate, report)
    
def endec_exec_1_step(self, epoch_count=10, batch_size=10, learning_rate=0.001, report=0, show_cnt=3):
    self.step(epoch_count, batch_size, learning_rate, report, show_cnt, 'both')
    self.test()
    if show_cnt > 0:
        self.visualize(show_cnt)
        
def endec_exec_2_step(self, epoch_count=10, batch_size=10, learning_rate=0.001, report=0, show_cnt=3):
    self.step(epoch_count, batch_size, learning_rate, report, 0, 'encoder')
    self.step(epoch_count, batch_size, learning_rate, report, show_cnt, 'decoder')
    self.set_train_mode('both')
    self.test()
    if show_cnt > 0:
        self.visualize(show_cnt)
        
def endec_exec_3_step(self, epoch_count=10, batch_size=10, learning_rate=0.001, report=0, show_cnt=3):
    self.step(epoch_count, batch_size, learning_rate, report, 0, 'encoder')
    self.step(epoch_count, batch_size, learning_rate, report, 0, 'dncoder')
    self.step(epoch_count, batch_size, learning_rate, report, show_cnt, 'both')
    self.test()
    if show_cnt > 0:
        self.visualize(show_cnt)
        
EncoderDecoder.set_train_mode = endec_set_train_mode
EncoderDecoder.step = endec_step
EncoderDecoder.exec_1_step = endec_exec_1_step
EncoderDecoder.exec_2_step = endec_exec_2_step
EncoderDecoder.exec_3_step = endec_exec_3_step

# 학습 모드별 알맞은 처리르 위한 순전파 메서드 재정의 
def endec_forward_neuralnet(self, x):
    hidden = x
    
    aux_encoder, aux_decoder = [], []
    
    if self.train_mode in ['both', 'encoder']:
        for n, hconfig in enumerate(self.econfigs):
            hidden, aux = self.forward_layer(hidden, hconfig, self.pm_encoder[n])
            aux_encoder.append(aux)
            
    if self.train_mode in ['both', 'decoder']:
        for n, hconfig in enumerate(self.dconfigs):
            hidden, aux = self.forward_layer(hidden, hconfig, self.pm_decoder[n])
            aux_decoder.append(aux)
            
    output = hidden
    
    return output, [aux_encoder, aux_decoder]

EncoderDecoder.forward_neuralnet = endec_forward_neuralnet

# 학습 모드별 알맞은 처리를 위한 역전파 메서드 재정의
def endec_backprop_neuralnet(self, G_output, aux):
    aux_encoder, aux_decoder = aux
    
    G_hidden = G_output
    
    if self.train_mode in ['both', 'decoder']:
        for n in reversed(range(len(self.dconfigs))):
            hconfig, pm = self.dconfigs[n], self.pm_decoder[n]
            aux = aux_decoder[n]
            G_hidden = self.backprop_layer(G_hidden, hconfig, pm, aux)
            
    if self.train_mode in ['both', 'encoder']:
        for n in reversed(range(len(self.econfigs))):
            hconfig, pm = self.econfigs[n], self.pm_encoder[n]
            aux = aux_encoder[n]
            G_hidden = self.backprop_layer(G_hidden, hconfig, pm, aux)
            
    return G_hidden 

EncoderDecoder.backprop_neuralnet = endec_backprop_neuralnet

# 콘텍스트 벡터 내용 확인을 위한 시각화 메서드 재정의 
def endec_visualize(self, num):
    print('Model {} Visualization'.format(self.name))
    self.set_train_mode('both')
    deX, deY = self.dataset.get_visualize_data(num)
    self.set_train_mode('encoder')
    code, _ = self.forward_neuralnet(deX)
    self.set_train_mode('decoder')
    output, _ = self.forward_neuralnet(code)
    self.dataset.visualize(deX, code, output, deY)
    
