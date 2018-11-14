# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:07:21 2018

@author: Yuki
"""

import numpy as np
import math as math


class MakeReceivedImg:
  def __init__(self, numberOfLEDs=9):
      """
      クラスの初期変数
      """
      self.numberOfLEDs = numberOfLEDs
      self.sqrtnumberofled = math.sqrt(numberOfLEDs)
    
  def RandomLEDs(self, max_lum_value=255, min_lum_value=0, gap=2):
    """
    Making randomly blinking LED and store them in array(numpy) 
    numberOfLEDs: Number of LEDs. Default = 16
    max_lum_value: maximum luminance value of LEDs
    min_lum_value: minimum luminance value of LEDs
    """
    
    
    # Generating random integers [0,1]
    blinking_leds = np.random.randint(0, 2, self.numberOfLEDs)

    # aplly max/min luminance value depending on their blinking condition
    for index, value in enumerate(blinking_leds):
      if value == 0:
        blinking_leds[index] = min_lum_value

      else:
        blinking_leds[index] = max_lum_value

    # lenで配列の長さを取得し、ルートする
    led_len = math.sqrt(len(blinking_leds))
    assert led_len % 1 == 0, 'LEDの個数がx^2の値でありません。例：4^2=16'

    # float型からint型へ変換
    led_len = int(led_len)

    # led_lenを一次元配列から二次元配列へ変換
    led_condition = np.reshape(blinking_leds, (led_len, led_len))

    return np.array(blinking_leds)
  
  
  def GaussChannelAndInv(self, sigma=0.4, kernelSize=5):
        """
        Creating channel parameters based on Gauss function
        sigma: 影響度の強さ
        kernelSize: ガウシアンフィルタのカーネルサイズ
        """
        
        y_scale = int(self.sqrtnumberofled)
        x_scale = int(self.sqrtnumberofled)        
        
        # 配列の確保
        gauss_channel = [[0 for i in range(y_scale * x_scale)] for j in range(y_scale * x_scale)]
        
        # 自身と他の画素から影響を受ける画素の決定
        for y in range(0, y_scale):
          for x in range(0, x_scale):
            target_pixel = y * x_scale + x  
        
            # target_pixelへ影響を与えるinfluence_pixelを決定
            for i in range(y_scale):
              for j in range(x_scale):
                
                influence_pixel = i * x_scale + j
                  
                # ガウシアンフィルの作成
                distance = (j - x) * (j - x) + (i - y) * (i - y)
                weight = math.exp(-distance / (2.0 * sigma * sigma))  * (1 / (2.0 * math.pi * sigma * sigma))
                
                if ((j - x) * (j - x)) >= kernelSize or ((i - y) * (i - y)) >= kernelSize:
                  weight = 0
               
                gauss_channel[target_pixel][influence_pixel] = weight
        
        gauss_channel = np.array(gauss_channel)
        
        # 逆行列の作成
        inv_channel = gauss_channel.copy()
        inv_gauss_channel = np.linalg.inv(inv_channel)
        return gauss_channel, inv_gauss_channel
      
      
  def Filtering(self, leds, sigma):
    """
    RandomLEDsで生成した配列にGaussChannelを施す
    """
    
    channel, invchannel = self.GaussChannelAndInv(sigma)
    #pixel_values = np.zeros(self.numberOfLEDs)
    
    pixel_values = np.dot(channel, leds)
    
   # pixel_values = np.dot(invchannel, pixel_values)
    
    return pixel_values
 
  """  
  def InvGaussChannel(self):
    
    #逆行列を施す
    
    
    channel = self.GaussChannel()
    inv_gauss_channel = np.linalg.inv(channel)
    return inv_gauss_channel
  """      
  
  
  def GetNoise(self, boxNoise=0.5):
    """
    ボックスミューラー法によってノイズを印加する
    """
    
    from scipy.stats import uniform
    
    # 独立した一様分布からそれぞれ一様乱数を生成
    np.random.seed()
    N = self.numberOfLEDs
    rv1 = uniform(loc=0.0, scale=1.0)
    rv2 = uniform(loc=0.0, scale=1.0)
    U1 = rv1.rvs(N)
    U2 = rv2.rvs(N)

    # Box-Mullerアルゴリズムで正規分布に従う乱数に変換
    # 2つの一様分布から2つの標準正規分布が得られる
    X1 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2) * boxNoise
    #X2 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2) * boxNoise
    
    return X1

if __name__ == '__main__':
   MakeReceivedImg() 