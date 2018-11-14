# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 21:47:09 2018

@author: Yuki
"""
import numpy as np
import pandas as pd

from dtreeviz.trees import dtreeviz
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score


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

df_train, df_train_target = DataFrameTrain(10000, 16, 2, 2, 0.1)
df_test, df_test_target = DataFrameTest(1000, 16, 2, 2, 0.1)

clf = tree.DecisionTreeClassifier(max_depth=4)  
clf.fit(df_train, df_train_target)

predicted = clf.predict(df_test)

print('=============== predicted ===============')
print(predicted)
print('============== correct_ans ==============')
print(sum(predicted == df_test_target) / len(df_test))