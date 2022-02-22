# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 20:49:29 2019

@author: admin
"""

#!/usr/bin/python
# -*- coding: UTF-8 -*-
import operator
import numpy as np
from numpy import *
from collections import Counter
'''
# introduction：
    The experimental data includes four attribute characteristics: age group, 
    whether there is work, whether there is a house, credit situation, 
    it is necessary to decide whether to give loans according to these 
    four attribute characteristics.
# 数据集
    train_data_set
    ID	AGE	    JOB	HOUSE	CREDIT	    GIVE LOANS
    1	youth	no	no	    general	        no
    2	youth	no	no	    good	        no
    3	youth	yes	no	    good	        yes
    4	youth	yes	yes	    general	        yes
    5	youth	no	no	    general	        no
    6	middle	no	no	    general	        no
    7	middle	no	no	    good	        no
    8	middle	yes	yes	    good	        yes
    9	middle	no	yes	    very good	    yes
    10	middle	no	yes	    very good	    yes
    11	old	    no	yes	    very good	    yes
    12	old	    no	yes	    good	        yes
    13	old	    yes	no	    good	        yes
    14	old	    yes	no	    very good	    yes
    15	old	    no	no	    general	        no
    16	old	    no	no	    very good	    no
# 数据集处理
    (0)	Age: 0 for youth, 1 for middle age, 2 for old age;
	(1)	There is work: 0 means no, 1 means yes;
	(2)	Have your own house: 0 means no, 1 means yes;
	(3)	Credit situation: 0 stands for general, 1 stands for good, 2 stands for very good;
	(4)	Category (whether to give loans): no means no, yes means yes.
'''

# 划分子集：根据特征（axis）的属性值（value）划分数据集，并返回等于属性值（value）的子集。
#解释equal：返回值等于或不等于value的子集，该值进行控制
def splitdataset(dataset,axis,value,isequal):       
    retdataset=[]                                   #定义一个用于存放子集的变量
    length=len(dataset)                             #元素个数获取，有几行
    if isequal:                                     #判断是否相等
        for i in range(length):                     #遍历
            if dataset[i][axis]==value:             #判断第一行的下标为axis的元素是不是等于value
                ret=dataset[i][:axis]               #是的话就将该元素前的所有元素给一个临时变量ret
                ret.extend(dataset[i][axis+1:])     #并且给这个临时变量ret添加上axis之后的元素，相当于不要axis列
                retdataset.append(ret)              #将新的一行元素添加到retdataset之中
    else:
        for i in range(length):
            if dataset[i][axis]!=value:
                ret=dataset[i][:axis]
                ret.extend(dataset[i][axis+1:])
                retdataset.append(ret)
    return retdataset                               #返回，这是一个部分，不是全部

# CART：根据基尼系数选择当前数据集的最优划分特征
def CART_chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1                           #特征的数量4
    numrows=len(dataset)                                        #行数，有多少个训练数据
    bestgini=0                                                  #因为计算的增益，最后取增益最大的，所以这里初始为0
    bestFeature = -1                                            #初始化最佳划分特征 -1
    array=mat(dataset)                                          #在CART_gini中有说明
    giniloan=CART_gini(dataset)
    for i in range(numFeatures):                                #循环
        count=Counter(np.transpose(array[:,i]).tolist()[0])
        axisnum=len(count)
        for item in count:
            isDataItem=splitdataset(dataset,i,item[0],True)     #取等于该元素的子集
            notDataItem=splitdataset(dataset,i,item[0],False)   #取不等于等于该元素的字集，因为是二分，所以必须这么做
            isDataItemGini=CART_gini(isDataItem)                #获取子集的基尼系数
            notDataItemGini=CART_gini(notDataItem)              #同上。#下面语句为计算基尼指数增益的公式，去百度。
            gini=giniloan-int(count[item])/numrows*isDataItemGini-(numrows-int(count[item]))/numrows*notDataItemGini
            if gini>bestgini:                                   #当增益出现新高度的时候，进行刷新
                bestgini=gini
                bestFeature=i
    return bestFeature                                          #返回一个值，是一个int类型的准确的值

# 返回一个dataset的基尼系数(1-something)的形式
def CART_gini(dataset):
    gini=0
    numrows=len(dataset)
    numFeatures = len(dataset[0]) - 1
    array=mat(dataset)                                              #将列表转化为矩阵
    dic=Counter(np.transpose(array[:,numFeatures]).tolist()[0])     #取矩阵的第numFeatures列，并转置成行，然后转化为列表，并放入字典dic之中，自动进行统计
    for item in dic:
        gini+=(dic[item]/numrows)**2                                #与下一行一起，是用于计算基尼系数的(基尼系数与基尼指数是两个不同的概念)
    return 1-gini

#CART决策树构建
def CART_createTree(dataset, labels):
    classList=[example[-1] for example in dataset]              #取分类标签(是否放贷：1(yes) or 0(no))
    if classList.count(classList[0])==len(classList):           #如果类别完全相同，则停止继续划分
        return classList[0]

    bestFeat=CART_chooseBestFeatureToSplit(dataset)             #选择最优特征
    bestFeatLabel=labels[bestFeat]                              #最优特征的标签
    CARTTree={bestFeatLabel:{}}                                 #根据最优特征的标签生成树
    del(labels[bestFeat])                                       #删除已经使用的特征标签
    featValues=[example[bestFeat] for example in dataset]       #得到训练集中所有最优特征的属性值
    uniqueVls=set(featValues)                                   #去掉重复的属性值

    for value in uniqueVls:                                     #遍历特征，创建决策树
        CARTTree[bestFeatLabel][value]=CART_createTree(splitdataset(dataset,bestFeat,value,True),labels)
    return CARTTree

#根据构建好的决策树以及对应的标签，对用例进行分类，输出分类结果0或1，这个函数是TireTree的搜索
def classify(inputTree, featLabels, testVec):
    firstStr=next(iter(inputTree))                                      #首先进入传进来的树的根节点，也就是house结点
    secondDict=inputTree[firstStr]                                      #然后定义一个字典，进入根节点的值空间之中，就是第二层花括号，看的时候很容易理解，此时花括号里面有两个元素，一个是确定的键值对，另一个是键-字典对
    featIndex=featLabels.index(firstStr)                                #根据传进的labels，判断这个根节点是第几列的
    for key in secondDict.keys():                                       #遍历这个字典，一般是有两对元素，一对是确定结果，另一个会进入深层的字典
        if testVec[featIndex]==key:                                     #如果说，对应列的测试数据等于这个键
            if type(secondDict[key]).__name__=='dict':                  #判断这个键是不是字典
                classLabel=classify(secondDict[key],featLabels,testVec) #如果是字典，就要进入递归
            else:
                classLabel=secondDict[key]                              #不是字典，就可以直接返回结果
    return classLabel                                                   #若以上都不是，就直接返回结果，这里返回的结果是一个准确的值

#主函数
if __name__ == '__main__':
    #数据集处理，这里的数据是已经处理好的，最后一列的0代表no，1代表yes
    dataset = [['0', '0', '0', '0', '0'], ['0', '0', '0', '1', '0'], ['0', '1', '0', '1', '1'],
               ['0', '1', '1', '0', '1'], ['0', '0', '0', '0', '0'], ['1', '0', '0', '0', '0'],
               ['1', '0', '0', '1', '0'], ['1', '1', '1', '1', '1'], ['1', '0', '1', '2', '1'],
               ['1', '0', '1', '2', '1'], ['2', '0', '1', '2', '1'], ['2', '0', '1', '1', '1'],
               ['2', '1', '0', '1', '1'], ['2', '1', '0', '2', '1'], ['2', '0', '0', '0', '0'],
               ['2', '0', '0', '2', '0']]
    labels = ['age', 'job', 'house', 'credit situation']    #label就是四个标签，构建决策树的时候需要使用
    labels_tmp = labels[:]                                  #因为在CART_createTree()函数里面会对于labels_tmp进行处理，所以这里拷贝了一个副本
    inidata='1,1,1,2'                                       #输入一条数据
    testVec=inidata.split(",")                              #分割
    # testVec=input().split(",")                            #在控制台进行输入时的语句，能够将input的东西分开
    CARTdesicionTree = CART_createTree(dataset, labels_tmp) #构建决策树
    print(CARTdesicionTree)
    print(classify(CARTdesicionTree, labels, testVec))      #输出最终的结果
    