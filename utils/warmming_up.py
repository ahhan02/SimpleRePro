#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: xmhan
@LastEditors: xmhan
@Date: 2019-04-11 23:31:06
@LastEditTime: 2019-04-18 12:13:50
'''
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def get_learning_rate(i, base_lr=0.001, burn_in=100, power=4):
    if i < burn_in:
        return base_lr * pow(i / burn_in, power)

base_lr = 0.001
lrs = []
for i in range(200):
    if i < 100:
        print(i, get_learning_rate(i))
        lrs.append(get_learning_rate(i))
    else:
        print(i, base_lr)
        lrs.append(base_lr)

plt.plot(range(200), lrs, color = 'r')
plt.show()

