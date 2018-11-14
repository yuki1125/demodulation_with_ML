# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:11:14 2018

@author: Yuki
"""

# main source
import numpy as np

from MakeReceivedImg import MakeReceivedImg

test = MakeReceivedImg(16)

sigma = 0.4
noise = 0.5
ch, invch = test.GaussChannelAndInv(sigma=sigma)

error_count = 0
iterations = 10000

print("Start processing....")
for iteration in range(iterations):
    leds = test.RandomLEDs(max_lum_value=1, min_lum_value=0)
    pixel = test.Filtering(leds, sigma=sigma)
    
    noise = test.GetNoise(noise)
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
    
    if iteration % 2000 == 0:
        print("現在は", (iteration / iterations) * 100, "%です")

print("================== ans accuracy ===================")
print("合計エラー個数は", error_count, "個でした")
print("BERは", error_count / (iterations * len(binary_data)))