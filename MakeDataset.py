# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 21:01:29 2018

@author: Yuki
"""
import numpy as np
import pandas as pd

from MakeReceivedImg import MakeReceivedImg

class MakeDataset(object):

    
    def __init__(self, loopLearn=100, loopTest=100, num_leds=16, noiseLevel=None, sigma=None):
        
        
        self.num_leds = num_leds
        self.sqrt_num = int(np.sqrt(num_leds))
        self.iteration_learn = loopLearn
        self.iteration_test = loopTest
        self.noise_level = noiseLevel
        self.sigma = sigma
       
        
    def GetPixelValue(self, maxLum=1, minLum=0):
        
        
        mri = MakeReceivedImg(numberOfLEDs=self.num_leds)
        store_led_condi = np.empty((0))
        store_px_value = np.empty((0))
        
        for iteration in range(self.iteration_learn):
            leds = mri.RandomLEDs(max_lum_value=maxLum, min_lum_value=minLum)
            store_led_condi = np.hstack((store_led_condi, leds))
            
            pixel = mri.Filtering(leds, self.sigma)
            noise = mri.GetNoise(self.noise_level)
            pixel_with_noise = pixel + noise
            store_px_value = np.hstack((store_px_value, pixel_with_noise))
        
        
        led_condi_2d = np.reshape(store_led_condi, (self.iteration_learn, self.num_leds))
        px_value_2d = np.reshape(store_px_value, (self.iteration_learn, self.num_leds))
        
        
        return px_value_2d, led_condi_2d
    
    
    def GetDataset(self, xTarget=None, yTarget=None):
        """
        xTarget: ターゲットとなるLEDのx座標
        yTarget: ターゲットとなるLEDのy座標
        データセットを作成する。
        GetPixelValueでピクセルバリューを取得
        この関数内で、ターゲットLEDを中心としたときの、ターゲットLEDと他LEDの距離を取得
        その２つをあわせたデータをデータ・セットとしてる
        """
        
        distance = np.empty((0))
        px_value_2d, led_condi_2d = self.GetPixelValue()
        tmp = int(yTarget * self.sqrt_num + xTarget)

        
        for i in range(0, self.iteration_learn):                
            for y in range(self.sqrt_num):
                y_dis = abs(yTarget - y)
                for x in range(self.sqrt_num):
                    x_dis = abs(xTarget - x)
                    euclid_distance = np.sqrt((y_dis) * (y_dis) + (x_dis) * (x_dis))
                    distance = np.hstack((distance, euclid_distance))
            
            onff = int(led_condi_2d[i][tmp])
            distance = np.hstack((distance, onff))
            
        
        distance_2d = np.reshape(distance, (self.iteration_learn, self.num_leds + 1))
        data = np.concatenate((px_value_2d, distance_2d), axis=1)
        df = pd.DataFrame(data, dtype='float')
        
        
        return df
    
if __name__ == "__main__":
    MakeDataset()
        