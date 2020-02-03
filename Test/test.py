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
