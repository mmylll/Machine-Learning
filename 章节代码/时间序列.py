# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:57:35 2021

@author: admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numpy.random import randn

data=randn(90)  #data可以替换成时间的时间序列变量
data=pd.Series(data)
data.index=pd.Index(sm.tsa.datetools.dates_from_range('2001','2090'))
data.plot(figsize=(14,7))
fig = plt.figure(figsize=(14,7))
diff_1 = data.diff(1)  #一阶差分
#自相关图和偏自相关图
fig = sm.graphics.tsa.plot_acf(data,lags=40,ax=fig.add_subplot(211))
fig = sm.graphics.tsa.plot_pacf(data,lags=40,ax=fig.add_subplot(212))
#ARMA(7,0)模型
arma_mod= sm.tsa.ARMA(data,(7,0)).fit()
#计算模型的AIC，BIC和HQIC
print(arma_mod.aic,arma_mod.bic,arma_mod.hqic)
predict_values=arma_mod.predict('2090', '2100', dynamic=True)
fig, ax=plt.subplots(figsize=(14,7))
ax=data.ix['2001':].plot(ax=ax)
predict_values.plot(ax=ax)

