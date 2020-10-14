# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 17:05:02 2020

@author: Eric
"""

import numpy as np 
from scipy.stats import norm
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
import math


class Learned_Index():
    
    def __init__(self, model_num):
        self.RMI = [1, model_num]    # Learned Index RMI架構
        self.index = []              # 儲存模型的索引
        self.N = None                # key值總數
        self.data = None             # 所有key值
        self.error_bound = []        # 儲存最後一層每個Model的 min_err and max_err 
        self.mean = None             # 儲存均值，資料標準化用
        self.std = None              # 儲存標準差，資料標準化用
        self.build()


    def build(self):
        for m in self.RMI:
            if m==1 :
                
                # 第一層 => 建置 NN Model 8x8
                model = Sequential()
                model.add(Dense(8, input_dim=1, activation="relu"))
                model.add(Dense(8, activation="relu"))
                model.add(Dense(1))
                
                sgd=keras.optimizers.SGD(lr=0.000001)    # lr:學習率,可調參數
                model.compile(loss="mse", optimizer=sgd, metrics=["mse"])
                self.index.append(model)
            else:
                # 第二層 => 建置多個簡單線性回歸
                self.index.append([])
                for j in range(m):
                    model = LinearRegression()
                    self.index[1].append(model)
        
    def train(self, data):

        self.data = data
        self.N = data.size
        y = self.crtCDF(data)
        norm_data = preprocessing.scale(data)    # 標準化: 零均值化
        self.mean = data.mean()
        self.std = data.std()
        # print("scale:",norm_data)
        # print("mine:",(data - self.mean)/data.std())

        for m in self.RMI:
            if m==1 :
                # 訓練第一層 NN 8x8 Model
                sgd=keras.optimizers.SGD(lr=0.000001)    # lr:學習率,可調參數
                self.index[0].compile(loss="mse", optimizer=sgd, metrics=["mse"])
                self.index[0].fit(norm_data, y, epochs=100, batch_size=32, verbose=0)

            else:
                # 依據第一層Model訓練結果將資料分配至第二層
                sub_data = [ [] for i in range(m)]        # 儲存第二層模型的各個keys
                sub_y = [ [] for i in range(m)]           # 儲存第二層模型的各個labels

                for i in range(self.N):
                    print(data[i])
                    mm = int(self.index[0].predict([[norm_data[i]]])*m/self.N)
                    
                    if mm < 0:
                        mm=0
                    elif mm > m-1:
                        mm = m-1

                    sub_data[mm].append(data[i])
                    sub_y[mm].append(y[i])

                # 訓練第二層所有的 SLR Model
                for j in range(m):
                    xx = np.array(sub_data[j])
                    yy = np.array(sub_y[j])

                    if xx.size > 0:
                        xx = np.reshape(xx,(-1,1))
                        self.index[1][j].fit(xx, yy)
                        

                # 計算最後一層 Model 的 min_err/max_err
                min_err = max_err = 0
                for i in range(data.size):
                    pred_pos, _ = self.predict(data[i])
                    err = pred_pos - i
                    if err < min_err:
                        min_err = math.floor(err)
                    elif err > max_err:
                        max_err = math.ceil(err)
                self.error_bound.append([min_err, max_err])

                
    def predict(self, key):
        mm = int(self.index[0].predict([[(key-self.mean)/self.std]])*self.RMI[1]/self.N)
        if mm < 0:
            mm=0
        elif mm > self.RMI[1]-1:
            mm = self.RMI[1]-1
        pred_pos = int(self.index[1][mm].predict([[key]]))
        return pred_pos, mm
        

    def search(self, key):   # model biased search
        pos, model = self.predict(key)

        l = pos + self.error_bound[model][0]
        r = pos + self.error_bound[model][1]



        # 檢查預測出的位置是否超出資料範圍
        if pos < 0:
            l = pos = 0 
        if pos > self.N-1:
            r = pos = self.N-1

        if l < 0:
            l=0
        if r > self.N-1:
            r = self.N-1

        # print(l,pos,r)

        while l<=r:

            if self.data[pos] == key:
                return True
            elif self.data[pos] > key:
                r = pos - 1
            elif self.data[pos] < key:
                l = pos + 1
            pos = int((l+r)/2)

        return False


    def crtCDF(self,x):
        if(type(x) == np.ndarray):
            loc = x.mean()
            scale = x.std()
            N = x.size
            pos = norm.cdf(x, loc, scale)*N
            return pos
        else:
            print("Wrong Type! x must be np.ndarray ~")    
            return


def main():

    data = np.array([1,4,5,10,23,31,48,51,67,80])
    li = Learned_Index(3)
    li.train(data)
    
    for k in data:
        print(li.search(k))

if __name__ == "__main__":
    main()
