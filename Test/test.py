# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:32:43 2020

@author: Jinsung
"""

#Chap1 test
runfile('C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap1-Regression analysis Estimation of the number of rings in abalone/abalone.py',
        wdir='C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap1-Regression analysis Estimation of the number of rings in abalone')
abalone_exec()

#Chap2 test
runfile('C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap2-Binary Classification predicting a pulsar star/pulsar.py',
        wdir='C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap2-Binary Classification predicting a pulsar star')
pulsar_exec()

runfile('C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap2-Binary Classification predicting a pulsar star/pulsar_ext.py',
        wdir='C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap2-Binary Classification predicting a pulsar star')
pulsar_exec(adjust_ratio=True)

#Chap3 test
runfile('C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap3-Multi Classification/steel_test.py',
        wdir='C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap3-Multi Classification')
steel_exec()

#Chap4 test
runfile('C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap4-MLP based structure/mlp.py',
        wdir='C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap4-MLP based structure')
runfile('C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap1-Regression analysis Estimation of the number of rings in abalone/abalone.py',
        wdir='C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap1-Regression analysis Estimation of the number of rings in abalone')
set_hidden([])
abalone_exec(epoch_count=50, report=10)

runfile('C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap4-MLP based structure/mlp.py',
        wdir='C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap4-MLP based structure')
runfile('C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap2-Binary Classification predicting a pulsar star/pulsar.py',
        wdir='C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap2-Binary Classification predicting a pulsar star')
set_hidden(6)
pulsar_exec(epoch_count=50, report=10)

set_hidden([12,6])
pulsar_exec(epoch_count=50, report=10)

runfile('C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap4-MLP based structure/mlp.py',
        wdir='C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap4-MLP based structure')
runfile('C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap2-Binary Classification predicting a pulsar star/pulsar_ext.py',
        wdir='C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap2-Binary Classification predicting a pulsar star')
set_hidden([12,6])
pulsar_exec(epoch_count=50, report=10, adjust_ratio=True)

runfile('C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap4-MLP based structure/mlp.py',
        wdir='C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap4-MLP based structure')
runfile('C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap3-Multi Classification/steel_test.py',
        wdir='C:/Users/Jinsung/Documents/Deep_Learning_Code/Chap3-Multi Classification')
LEARNING_RATE = 0.0001
set_hidden([12,6,4])
steel_exec(epoch_count=50, report=10)

#Chap5 test
ad = AbaloneDataset()
am = MlpModel('abalone_model',ad,[])
am.exec_all(epoch_count=10, report=2)

pd = PulsarDataset()
pm = MlpModel('pulsar_model', pd, [4])
pm.exec_all()
pm.visualize(5)

sd = SteelDataset()
sm = MlpModel('steel_model', sd, [12,7])
sm.exec_all(epoch_count=50, report=10)

psd = PulsarSelectDataset()
psm = MlpModel('pulsar_select_model', psd, [4])
psm.exec_all()

fd = FlowersDataset()
fm = MlpModel('flowers_model_1', fd, [10]) #같은 추정확률분포를 통해 언제나 민들레라는 답을 냈다.
fm.exec_all(epoch_count=10, report=2)
fm2 = MlpModel('flowers_model_2', fd, [30,10])
fm2.exec_all(epoch_count=10, report=2)

#Chap6 test
od = Office31Dataset()
om1 = MlpModel('office31_model_1', od, [10])
om1.exec_all(epoch_count=20, report=10)


om2 = MlpModel('office31_model_2', od, [64,32,10])
om2.exec_all(epoch_count=20, report=10, learning_rate=0.0001)

om3 = MlpModel('office31_model_3', od, [64,32,10])
om3.use_adam = True
om3.exec_all(epoch_count=50, report=10, learning_rate=0.0001)

#Chap7 test
fd = FlowersDataset([96, 96], [96, 96, 3])
od = Office31Dataset([96, 96], [96, 96, 3])

fm1 = CnnBasicModel('flowers_model_1', fd, [30, 10])
fm1.exec_all(epoch_count=10, report=2)

fm2 = CnnBasicModel('flowers_model_2', fd, [['full', {'width':30}],['full', {'width':10}]])
fm2.use_adam=False
fm2.exec_all(epoch_count = 10, report = 2)

fm3 = CnnBasicModel('flowers_model_3', fd, [['conv', {'ksize':5, 'chn':6}],
                                            ['max', {'stride':4}], 
                                            ['conv', {'ksize':3, 'chn':12}],
                                            ['avg', {'stride':2}]], 
                                            True)
fm3.exec_all(epoch_count = 10, report=2)

om1 = CnnBasicModel('officie31_model_1', od, 
                    [['conv', {'ksize':3, 'chn':6}],
                     ['max', {'stride':2}], 
                     ['conv', {'ksize':3, 'chn':12}],
                     ['max', {'stride':2}],
                     ['conv', {'ksize':3, 'chn':24}],
                     ['avg', {'stride':3}]])
om1.exec_all(epoch_count=10, report =2)

om2 = CnnBasicModel('officie31_model_2', od, 
                    [['conv', {'ksize':3, 'chn':6, 'actfunc':'sigmoid'}],
                     ['max', {'stride':2}], 
                     ['conv', {'ksize':3, 'chn':12, 'actfunc':'sigmoid'}],
                     ['max', {'stride':2}],
                     ['conv', {'ksize':3, 'chn':24, 'actfunc':'sigmoid'}],
                     ['avg', {'stride':3}]])
om2.exec_all(epoch_count=10, report =2)

om3 = CnnBasicModel('officie31_model_3', od, 
                    [['conv', {'ksize':3, 'chn':6, 'actfunc':'tanh'}],
                     ['max', {'stride':2}], 
                     ['conv', {'ksize':3, 'chn':12, 'actfunc':'tanh'}],
                     ['max', {'stride':2}],
                     ['conv', {'ksize':3, 'chn':24, 'actfunc':'tanh'}],
                     ['avg', {'stride':3}]])
om3.exec_all(epoch_count=10, report =2)

#Chap8 test
fd = FlowersDataset([96, 96], [96, 96, 3])
od = Office31Dataset([96, 96], [96, 96, 3])
fm1 = CnnRegModel('flowers_model_1', fd, [30, 10])
fm1.exec_all(epoch_count=10, report=2, show_params=True)

fm2 = CnnRegModel('flowers_model_2', fd, [30, 10], l2_decay=0.1)
fm2.exec_all(epoch_count=10, show_cnt=0, show_params=True)

fm3 = CnnRegModel('flowers_model_3', fd, [30, 10], l1_decay=0.1)
fm3.exec_all(epoch_count=10, show_cnt=0, show_params=True)

cnn1 = [['conv', {'ksize':3, 'chn':6}],
        ['max', {'stride':2}], 
        ['conv', {'ksize':3, 'chn':12}],
        ['max', {'stride':2}],
        ['conv', {'ksize':3, 'chn':24}],
        ['avg', {'stride':3}]]
fcnn1 = CnnRegModel('flowers_cnn_1')
fcnn1.exec_all(epoch_count=10, report=2)

cnn2 = [['conv', {'ksize':3, 'chn':6}],
        ['max', {'stride':2}], 
        ['dropout', {'keep_prob':0.6}],
        ['conv', {'ksize':3, 'chn':12}],
        ['max', {'stride':2}],
        ['dropout', {'keep_prob':0.6}],
        ['conv', {'ksize':3, 'chn':24}],
        ['avg', {'stride':3}],
        ['dropout', {'keep_prob':0.6}]]

fcnn2 = CnnRegModel('flowers_cnn_2',fd, cnn2)
fcnn2.exec_all(epoch_count=10, report=2, show_cnt=0)

cnn3 = [['noise', {'type':'normal','mean':0,'std':0.01}],
        ['conv', {'ksize':3, 'chn':6}],
        ['max', {'stride':2}], 
        ['noise', {'type':'normal','mean':0,'std':0.01}],
        ['conv', {'ksize':3, 'chn':12}],
        ['max', {'stride':2}],
        ['noise', {'type':'normal','mean':0,'std':0.01}],
        ['conv', {'ksize':3, 'chn':24}],
        ['avg', {'stride':3}]]

fcnn3 = CnnRegModel('flowers_cnn_3',fd, cnn3)
fcnn3.exec_all(epoch_count=10, report=2, show_cnt=0)

cnn4 = [['batch_normal'],
        ['conv', {'ksize':3, 'chn':6}],
        ['max', {'stride':2}], 
        ['batch_normal'],
        ['conv', {'ksize':3, 'chn':12}],
        ['max', {'stride':2}],
        ['batch_normal'],
        ['conv', {'ksize':3, 'chn':24}],
        ['avg', {'stride':3}]]

fcnn4 = CnnRegModel('flowers_cnn_4',fd, cnn4)
fcnn4.exec_all(epoch_count=10, report=2, show_cnt=0)

od = Office31Dataset([96, 96], [96, 96, 3])
ocnn1 = CnnRegModel('office31_cnn_1', od, cnn1)
ocnn2 = CnnRegModel('office31_cnn_2', od, cnn1)
ocnn3 = CnnRegModel('office31_cnn_3', od, cnn1)
ocnn4 = CnnRegModel('office31_cnn_4', od, cnn1)

ocnn1.exec_all(epoch_count=10, show_cnt=0)
ocnn2.exec_all(epoch_count=10, show_cnt=0)
ocnn3.exec_all(epoch_count=10, show_cnt=0)
ocnn4.exec_all(epoch_count=10, show_cnt=0)

# Chap9
# inception-v3  model
imagenet = DummyDataset('imagenet', 'select', [299,299,3], 200)
CnnExtModel.set_macro('v3_preproc',
                      ['serial',
                       ['conv', {'ksize':3, 'stride':2, 'chn':32, 'padding':'VALID'}],
                       ['conv', {'ksize':3, 'chn':32, 'padding':'VALID'}],
                       ['conv', {'ksize':3, 'chn':64, 'padding':'SAME'}],
                       ['max', {'ksize':3, 'stride':2, 'padding':'VALID'}],
                       ['conv', {'ksize':1, 'chn':80, 'padding':'VALID'}],
                       ['max', {'ksize':3, 'stride':2, 'padding':'VALID'}]])
CnnExtModel.set_macro('v3_inception1',
                      ['parallel',
                       ['conv', {'ksize':1, 'chn':64}],
                       ['serial',
                        ['conv', {'ksize':1, 'chn':48}],
                        ['conv', {'ksize':5, 'chn':64}]],
                       ['serial',
                        ['conv', {'ksize':1, 'chn':64}],
                        ['conv', {'ksize':3, 'chn':96}],
                        ['conv', {'ksize':3, 'chn':96}]],
                       ['serial',
                        ['avg', {'ksize':3, 'stride':1}],
                        ['conv', {'ksize':1, 'chn':'#chn'}]]])
CnnExtModel.set_macro('v3_resize1',
                      ['parallel',
                       ['conv', {'ksize':1, 'stride':2, 'chn':384}],
                       ['serial',
                        ['conv', {'ksize':1, 'chn':64}],
                        ['conv', {'ksize':3, 'chn':96}],
                        ['conv', {'ksize':3, 'stride':2, 'chn':96}]],
                       ['max', {'ksize':3, 'stride':2}]])
CnnExtModel.set_macro('v3_inception2',
                      ['parallel',
                       ['conv', {'ksize':1, 'chn':192}],
                       ['serial',
                        ['conv', {'ksize':[1,1], 'chn':'#chn'}],
                        ['conv', {'ksize':[1,7], 'chn':'#chn'}],
                        ['conv', {'ksize':[7,1], 'chn':192}]],
                       ['serial',
                        ['conv', {'ksize':[1,1], 'chn':'#chn'}],
                        ['conv', {'ksize':[7,1], 'chn':'#chn'}],
                        ['conv', {'ksize':[1,7], 'chn':'#chn'}],
                        ['conv', {'ksize':[7,1], 'chn':'#chn'}],
                        ['conv', {'ksize':[1,7], 'chn':192}]],
                       ['serial',
                        ['avg', {'ksize':3, 'stride':1}],
                        ['conv', {'ksize':1, 'chn':192}]]])
CnnExtModel.set_macro('v3_resize2',
                      ['parallel',
                       ['serial',
                        ['conv', {'ksize':1, 'chn':192}],
                        ['conv', {'ksize':3, 'stride':2, 'chn':320}]],
                       ['serial',
                        ['conv', {'ksize':[1,1], 'chn':192}],
                        ['conv', {'ksize':[1,7], 'chn':192}],
                        ['conv', {'ksize':[7,1], 'chn':192}],
                        ['conv', {'ksize':[3,3], 'stride':[2,2], 'chn':192}]],
                       ['max', {'ksize':3, 'stride':2}]])
CnnExtModel.set_macro('v3_inception3',
                      ['parallel',
                       ['conv', {'ksize':1, 'chn':320}],
                       ['serial',
                        ['conv', {'ksize':[3,3], 'chn':384}],
                        ['parallel',
                         ['conv', {'ksize':[1,3], 'chn':384}],
                         ['conv', {'ksize':[3,1], 'chn':384}]]],
                       ['serial',
                        ['conv', {'ksize':[1,1], 'chn':448}],
                        ['conv', {'ksize':[3,3], 'chn':384}],
                        ['parallel',
                         ['conv', {'ksize':[1,3], 'chn':384}],
                         ['conv', {'ksize':[3,1], 'chn':384}]]],
                       ['serial',
                        ['avg', {'ksize':3, 'stride':1}],
                        ['conv', {'ksize':1, 'chn':192}]]])
CnnExtModel.set_macro('v3_postproc',
                      ['serial',
                       ['avg', {'stride':8}],
                       ['dropout', {'keep_prob':0.7}]])
CnnExtModel.set_macro('inception_v3',
                      ['serial',
                       ['custom', {'name':'v3_preproc'}],
                       ['custom', {'name':'v3_inception1', 'args':{'#chn':32}}],
                       ['custom', {'name':'v3_inception1', 'args':{'#chn':64}}],
                       ['custom', {'name':'v3_inception1', 'args':{'#chn':64}}],
                       ['custom', {'name':'v3_resize1'}],
                       ['custom', {'name':'v3_inception2', 'args':{'#chn':128}}],
                       ['custom', {'name':'v3_inception2', 'args':{'#chn':160}}],
                       ['custom', {'name':'v3_inception2', 'args':{'#chn':160}}],
                       ['custom', {'name':'v3_inception2', 'args':{'#chn':192}}],
                       ['custom', {'name':'v3_resize2'}],
                       ['custom', {'name':'v3_inception3'}],
                       ['custom', {'name':'v3_inception3'}],
                       ['custom', {'name':'v3_postproc'}]])

inception_v3 = CnnExtModel('inception_v3', imagenet, [['custom', {'name':'inception_v3'}]], dump_structure=True)

fd = FlowersDataset([96, 96], [96, 96, 3])
CnnExtModel.set_macro('flower_preproc',
                      ['serial',
                       ['conv', {'ksize':3, 'stride':2, 'chn':6, 'actions':'#act'}]])
CnnExtModel.set_macro('flower_inception1',
                      ['parallel',
                       ['conv', {'ksize':1, 'chn':4, 'actions':'#act'}],
                       ['conv', {'ksize':3, 'chn':6, 'actions':'#act'}],
                       ['serial',
                        ['conv', {'ksize':3, 'chn':6, 'actions':'#act'}],
                        ['conv', {'ksize':3, 'chn':6, 'actions':'#act'}]],
                       ['serial',
                        ['avg', {'ksize':3, 'stride':1}],
                        ['conv', {'ksize':1, 'chn':4, 'actions':'#act'}]]])
CnnExtModel.set_macro('flower_resize',
                      ['parallel',
                       ['conv', {'ksize':1, 'stride':2, 'chn':12, 'actions':'#act'}],
                       ['serial',
                        ['conv', {'ksize':3, 'chn':12, 'actions':'#act'}],
                        ['conv', {'ksize':3, 'stride':2, 'chn':12, 'actions':'#act'}]],
                       ['avg', {'ksize':3, 'stride':2}]])
CnnExtModel.set_macro('flower_inception2',
                      ['parallel',
                       ['conv', {'ksize':1, 'chn':8, 'action':'#act'}],
                       ['serial',
                        ['conv', {'ksize':[3,3], 'chn':8, 'actions':'#act'}],
                        ['parallel',
                         ['conv', {'ksize':[1,3], 'chn':8, 'actions':'#act'}],
                         ['conv', {'ksize':[3,1], 'chn':8, 'actions':'#act'}]]],
                       ['serial',
                        ['conv', {'ksize':[1,1], 'chn':8, 'actions':'#act'}],
                        ['conv', {'ksize':[3,3], 'chn':8, 'actions':'#act'}],
                        ['parallel',
                         ['conv', {'ksize':[1,3], 'chn':8, 'actions':'#act'}],
                         ['conv', {'ksize':[3,1], 'chn':8, 'actions':'#act'}]]],
                       ['serial',
                        ['avg', {'ksize':3, 'stride':1}],
                        ['conv', {'ksize':1, 'chn':8, 'actions':'#act'}]]])
CnnExtModel.set_macro('flower_postproc',
                      ['serial',
                       ['avg', {'stride':6}],
                       ['dropout', {'keep_prob':0.7}]])
CnnExtModel.set_macro('inception_flower',
                      ['serial',
                       ['custom', {'name':'flower_preproc', 'args':{'#act':'#act'}}],
                       ['custom', {'name':'flower_inception1', 'args':{'#act':'#act'}}],
                       ['custom', {'name':'flower_resize', 'args':{'#act':'#act'}}],
                       ['custom', {'name':'flower_inception1', 'args':{'#act':'#act'}}],
                       ['custom', {'name':'flower_resize', 'args':{'#act':'#act'}}],
                       ['custom', {'name':'flower_inception2', 'args':{'#act':'#act'}}],
                       ['custom', {'name':'flower_resize', 'args':{'#act':'#act'}}],
                       ['custom', {'name':'flower_inception2', 'args':{'#act':'#act'}}],
                       ['custom', {'name':'flower_postproc', 'args':{'#act':'#act'}}]])

conf_flower_LA = ['custom', {'name':'inception_flower', 'args':{'#act':'LA'}}]
model_flower_LA = CnnExtModel('model_flower_LA', fd, conf_flower_LA, dump_structure=True)

model_flower_LA.exec_all(report=2)

conf_flower_LAB = ['custom', {'name':'inception_flower', 'args':{'#act':'LAB'}}]
model_flower_LAB = CnnExtModel('model_flower_LAB', fd, conf_flower_LAB, dump_structure=False)
model_flower_LAB.exec_all(epoch_count=10, report=2)

#Chap10
ad = AutomataDataset()

am_4 = RnnBasicModel('am_4', ad, ['rnn', {'recur_size':4, 'outseq':False}])
am_16 = RnnBasicModel('am_16', ad, ['rnn', {'recur_size':16, 'outseq':False}])
am_64 = RnnBasicModel('am_64', ad, ['rnn', {'recur_size':64, 'outseq':False}])

am_4.exec_all(epoch_count=10, report=2)
am_16.exec_all(epoch_count=10, report=2)
am_64.exec_all(epoch_count=10, report=2)

am_64_drop = RnnBasicModel('am_64_drop', ad, [['rnn', {'recur_size':64, 'outseq':False}],['dropout', {'keep_prob':0.5}]])
am_64_drop.exec_all(epch_count=10, report=2)

#Chap11
ad = AutomataDataset()
am_4 = RnnLstmModel('am_4', ad, ['lstm', {'recur_size':64, 'outseq':False}])
am_4.exec_all(epoch_count=10, report=2)

usd_10_10 = UrbanSoundDataset(10, 10)
usd_10_100 = UrbanSoundDataset(10, 100)

conf_basic = ['rnn', {'recur_size':20, 'outseq':False}]
conf_lstm = ['lstm', {'recur_size':20, 'outseq':False}]
conf_state = ['lstm', {'recur_size':20, 'outseq':False, 'use_state':True}]

us_basic_10_10 = RnnLstmModel('us_basic_10_10', usd_10_10, conf_basic)
us_lstm_10_10 = RnnLstmModel('us_lstm_10_10', usd_10_10, conf_lstm)
us_state_10_10 = RnnLstmModel('us_state_10_10', usd_10_10, conf_state)

us_basic_10_100 = RnnLstmModel('us_basic_10_100', usd_10_100, conf_basic)
us_lstm_10_100 = RnnLstmModel('us_lstm_10_100', usd_10_100, conf_lstm)
us_state_10_100 = RnnLstmModel('us_state_10_100', usd_10_100, conf_state)

us_basic_10_10.exec_all(epoch_count=10, report=2)
us_lstm_10_10.exec_all(epoch_count=10, report=2)
us_state_10_10.exec_all(epoch_count=10, report=2, show_cnt=0)

#Chap12
vsd = np.load('C:\\Users\\Jinsung\\Documents\\Deep_Learning_Code\\Datasets\\chap12\\cache\\AstarIsBorn1937.mp4.npy')

conf1 = [['seqwrap', ['avg', {'stride':30}],
                     ['conv', {'ksize':3, 'chn':12}],
                     ['full', {'width':16}]],
         ['lstm', {'recur_size':8}]]
vsm1 = RnnExtModel('vsm1', vsd, conf1)
vsm1.exec_all(epoch_count=10, report=2, show_cnt=3)
vsd.shape

#Chap13
mset_all = MnistAutoDataset(train_ratio=1.00)
mset_1p = MnistAutoDataset(train_ratio=0.01)

conf_mlp = [['full',{'width':10}]]
mnist_mlp_all = RnnExtModel('mnist_mlp_all', mset_all, conf_mlp)
mnist_mlp_all.exec_all(epoch_count=10, report=2)

conf_auto = {
    'encoder': [['full', {'width':10}]],
    'decoder': [['full', {'width':784}]],
    'supervised': [['full', {'width':10}]]
    }

mnist_auto_1 = Autoencoder('mnist_auto_1',mset_1p, conf_auto)
mnist_auto_1.autoencode(epoch_count=10, report=2)
mnist_auto_1.exec_all(epoch_count=10, report=2)


mnist_auto_fix = Autoencoder('mnist_auto_fix', mset_1p, conf_auto, fix_encoder=True)
mnist_auto_fix.autoencode(epoch_count=10, report=5)
mnist_auto_fix.exec_all(epoch_count=10, report=5)

conf_auto_2 = {
    'encoder': [['full', {'width':64}], ['full', {'width':10}]],
    'decoder': [['full', {'width':64}], ['full', {'width':784}]],
    'supervised': [['full', {'width':10}]]
    }

mnist_auto_2 = Autoencoder('mnist_auto_2',mset_1p, conf_auto_2)
mnist_auto_2.autoencode(epoch_count=10, report=2)
mnist_auto_2.exec_all(epoch_count=10, report=2)

conf_hash_1 = {
    'encoder': [['full', {'width':10, 'actfunc':'sigmoid'}]],
    'decoder': [['full', {'width':784}]],
    'supervised': []
    }
mnist_hash_1 = Autoencoder('mnist_hash_1',mset_1p, conf_hash_1)
mnist_hash_1.autoencode(epoch_count=10, report=2)
mnist_hash_1.semantic_hashing_index()
mnist_hash_1.semantic_hashing_search()

conf_hash_2 = {
    'encoder': [['full', {'width':64}],['full', {'width':10, 'actfunc':'sigmoid'}]],
    'decoder': [['full', {'width':64}],['full', {'width':784}]],
    'supervised': []
    }
mnist_hash_2 = Autoencoder('mnist_hash_2',mset_1p, conf_hash_2)
mnist_hash_2.autoencode(epoch_count=10, report=2)
mnist_hash_2.semantic_hashing_index()
mnist_hash_2.semantic_hashing_search()

mnist_hash_2.autoencode(epoch_count=40, report=10)
mnist_hash_2.semantic_hashing_index()
mnist_hash_2.semantic_hashing_search()

