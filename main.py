# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:11:14 2018

@author: Yuki
"""

# main source
import numpy as np

from MakeReceivedImg import MakeReceivedImg

test = MakeReceivedImg()

ch, invch = test.GaussChannelAndInv()

error_count = 0
iteration = 100

for iteration in range(iteration):
    leds = test.RandomLEDs(max_lum_value=1, min_lum_value=0)
    pixel, leds = test.Filtering(leds)
    
    noise = test.GetNoise(0.3)
    pixel_with_noise = pixel + noise 
    estimation = np.dot(invch, pixel_with_noise)
    
    binary_data = []
    
    for thr_est in estimation:
        if thr_est > 0.5:
            binary_data.append(1)
        
        else:
            binary_data.append(0)
    
    for i in binary_data:
        if not leds[i] == binary_data[i]:
            error_count += 1
            

print("合計エラー個数は", error_count, "個でした")
print("BERは", error_count / (iteration * len(binary_data)))