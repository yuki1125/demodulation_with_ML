# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 21:47:09 2018

@author: Yuki
"""
import time

import numpy as np
import pandas as pd
 

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

#from MakeReceivedImg import MakeReceivedImg
from MakeDataset import MakeDataset


def DataFrameTrain(loopTimes=None, numLEDs=None, xTarget=None, yTarget=None, noiseLevel=None):

    
    df_train = MakeDataset(noiseLevel=noiseLevel, loopLearn=loopTimes, num_leds=numLEDs)
    data_train = df_train.GetDataset(xTarget=0, yTarget=0)
    row = int(numLEDs * 2)
    data_train_target = data_train.pop(row)
    
    #df_train_target = data_train[-1]
    return data_train, data_train_target

def DataFrameTest(loopTimes=None, numLEDs=None, xTarget=None, yTarget=None, noiseLevel=None):
    
    
    df_test = MakeDataset(noiseLevel=noiseLevel, loopLearn=loopTimes, num_leds=numLEDs)
    data_test = df_test.GetDataset(xTarget=0, yTarget=0)
    row = int(numLEDs * 2)
    data_test_target = data_test.pop(row)
    
    
    return data_test, data_test_target

led_num = 16
num_error = 0
sqrt_len = int(np.sqrt(led_num))

print('============== Start Training ==============')
start = time.time()
for y in range(sqrt_len):
    for x in range(sqrt_len):
        df_train, df_train_target = DataFrameTrain(1000, led_num, y, x, 0.2)
        df_test, df_test_target = DataFrameTest(1000, led_num, y, x, 0.2)
        
        clf = tree.DecisionTreeClassifier(max_depth=4)  
        clf.fit(df_train, df_train_target)
        
        predicted = clf.predict(df_test)
        num_error += (sum(predicted != df_test_target))
    
    elapsed_time = time.time() - start
    
    print('Training progress.....')
    print('現在: ', ((y + 1) * 100) / sqrt_len, '% です. (合計処理時間:', elapsed_time, '[sec])')

print('Complete Training')
#print('=============== predicted ===============')
#print(predicted)
print('============== correct_ans ==============')
print('BER:', num_error / (len(df_test) * led_num))