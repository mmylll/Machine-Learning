# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 13:46:00 2021

@author: mmy
"""

#-*- coding:utf-8 -*- 
 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
 
#读取文件
datafile = u'身高预测参照表-1.xlsx'#文件所在位置，u为防止路径中有中文名称
data = pd.read_excel(datafile)#datafile是excel文件，所以用read_excel,如果是csv文件则用read_csv
examDf = DataFrame(data)
examDf.head()
print(data)
 
#数据清洗,比如第一列可能用不到，这样的话我们就只需要从第二列开始的数据，
#这个情况下，把下面中括号中的0改为1就好，要哪些列取哪些列
new_examDf = examDf.iloc[:,1:]
 
#检验数据
print("--------------------------------------------------------------------")
print(u"数据描述")
print(new_examDf.describe())#数据描述，会显示最值，平均数等信息，可以简单判断数据中是否有异常值
print("--------------------------------------------------------------------")
print(u"检验缺失值")
print(new_examDf[new_examDf.isnull()==True].count())#检验缺失值，若输出为0，说明该列没有缺失值
print("--------------------------------------------------------------------")
#输出相关系数，判断是否值得做线性回归模型
print(u"相关系数")
print(new_examDf.corr())#0-0.3弱相关；0.3-0.6中相关；0.6-1强相关；


#通过seaborn添加一条最佳拟合直线和95%的置信带，直观判断相关关系
sns.pairplot(data, x_vars=['足长','步幅'], y_vars='身高', size=7, aspect=0.8, kind='reg')  
plt.show()

X_train,X_test,Y_train,Y_test = train_test_split(new_examDf.iloc[:,:2],new_examDf.身高,train_size=0.8)
#new_examDf.ix[:,:2]取了数据中的前两列为自变量，与单变量的不同
 
print(u"自变量---源数据:",new_examDf.iloc[:,:2].shape, u"；  训练集:",X_train.shape, u"；  测试集:",X_test.shape)
print(u"因变量---源数据:",examDf.身高.shape, u"；  训练集:",Y_train.shape, u"；  测试集:",Y_test.shape)


#调用线性规划包
model = LinearRegression()
 
model.fit(X_train,Y_train)#线性回归训练
 
a  = model.intercept_#截距
b = model.coef_#回归系数
print(u"拟合参数:截距",a,u",回归系数：",b)
 
#显示线性方程，并限制参数的小数位为十位
print(u"最佳拟合线: Y = ",round(a,10),"+",round(b[0],10),"* X1 + ",round(b[1],10),"* X2")
 
Y_pred = model.predict(X_test)#对测试集数据，用predict函数预测

# Mean Squared Error 均方误差
mse = np.average((Y_pred - Y_test) ** 2)
print(u'均方误差：', mse)
# 模型得分
print(u'模型得分:',model.score(X_test, Y_test))
 
plt.plot(range(len(Y_pred)),Y_pred,'red', linewidth=2.5,label="predict data")
plt.plot(range(len(Y_test)),Y_test,'green',label="test data")
plt.legend(loc=2)
plt.show()#显示预测值与测试值曲线