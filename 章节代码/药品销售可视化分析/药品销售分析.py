# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:30:04 2019

@author: admin
"""

import pandas as pd
 
# 读取数据（最好使用 object 类型读取）
data = pd.read_excel("朝阳医院2018年销售数据.xlsx", dtype="object")
 
 # 修改为 DataFrame 格式
dataDF = pd.DataFrame(data)
 
 # 查看数据的形状，即几行几列
print(dataDF.shape)

print(dataDF.index)
# 查看每一列的列表头内容
print(dataDF.columns)
 # 查看每一列数据统计数目
print(dataDF.count())
# 使用 rename 函数，把"购药时间" 改为 "销售时间"
dataDF.rename(columns={"购药时间": "销售时间"}, inplace=True)
print(dataDF.columns)
 # 删除缺失值之前
print(dataDF.shape)
# 使用dropna函数删除缺失值
dataDF = dataDF.dropna()
 # 删除缺失值之后
print(dataDF.shape)
# 将字符串转为浮点型数据
dataDF["销售数量"] = dataDF["销售数量"].astype("f8")
dataDF["应收金额"] = dataDF["应收金额"].astype("f8")
dataDF["实收金额"] = dataDF["实收金额"].astype("f8")
print(dataDF.dtypes)

def splitsaletime(timeColser):
  timelist = []
  for t in timeColser:
     # [0]表示选取的分片，这里表示切割完后选取第一个分片
    timelist.append(t.split(" ")[0])
     # 将列表转行为一维数据Series类型
    timeser = pd.Series(timelist)
  return timeser



# 获取"销售时间"这一列数据
t = dataDF.loc[:, "销售时间"]
# 调用函数去除星期，获取日期
timeser = splitsaletime(t)
# 修改"销售时间"这一列日期
dataDF.loc[:, "销售时间"] = timeser
print(dataDF.head())

# 字符串转日期
# errors='coerce'如果原始数据不符合日期的格式，转换后的值为NaT
dataDF.loc[:, "销售时间"] = pd.to_datetime(dataDF.loc[:, "销售时间"], errors='coerce')
print(dataDF.dtypes)

# 转换日期过程中不符合日期格式的数值会被转换为空值None，
 # 这里删除为空的行
dataDF = dataDF.dropna()
print(dataDF.shape)
# 按销售时间进行升序排序
dataDF = dataDF.sort_values(by='销售时间', ascending=True)
print(dataDF.head())
# 重置索引（index）
dataDF = dataDF.reset_index(drop=True)

# 查看描述统计信息
dataDF.describe()
# 将"销售数量"这一列中小于0的数排除掉
pop = dataDF.loc[:, "销售数量"] > 0
dataDF = dataDF.loc[pop, :]
 
# 排除异常值后再次查看描述统计信息
dataDF.describe()
# 计算总消费次数
# 删除重复数据
kpi1_Df = dataDF.drop_duplicates(subset=['销售时间', '社保卡号'])
 
# 删除重复数据
kpi1_Df = dataDF.drop_duplicates(subset=['销售时间', '社保卡号'])
 
# 有多少行
totall = kpi1_Df.shape[0]
print('总消费次数：', totall)
# 按销售时间升序排序
kpi1_Df = kpi1_Df.sort_values(by='销售时间', ascending=True)
 
# 重命名行名（index）
kpi1_Df = kpi1_Df.reset_index(drop=True)
 
# 获取时间范围
 # 最小时间值
startTime = kpi1_Df.loc[0, '销售时间']
# 最大时间值
endTime = kpi1_Df.loc[totall - 1, '销售时间']
 
# 计算天数
daysI = (endTime - startTime).days
 
# 月份数：运算符"//"表示取整除，返回商的整数部分
monthsI = daysI // 30
print('月份数：', monthsI)
# 计算月均消费次数
kpi1_I = totall // monthsI
print('业务指标1：月均消费次数=', kpi1_I)
# 总消费金额
totalMoneyF = dataDF.loc[:, '实收金额'].sum()
 
# 月均消费金额
monthMoneyF = totalMoneyF / monthsI
print('业务指标2：月均消费金额=', monthMoneyF)
 # 客单价 = 总消费金额 / 总消费次数
pct = totalMoneyF / totall
print('业务指标3：客单价=', pct)
import matplotlib.pyplot as plt
 # 画图时用于显示中文字符
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']   # SimHei是黑体的意思
 
# 在操作之前先复制一份数据，防止影响清洗后的数据
groupDf = dataDF
 
# 重命名行（index）为销售时间所在列的值
groupDf.index = groupDf['销售时间']
groupDf.head()
# 画图
plt.plot(groupDf['实收金额'])
plt.title('按天消费金额图')
plt.xlabel('时间')
plt.ylabel('实收金额')
 # 保存图片
plt.savefig('./day.png')
# 显示图片
plt.show()
 
# 将销售时间聚合按月分组
gb = groupDf.groupby(groupDf.index.month)
# 应用函数，计算每个月的消费总额
monthDf = gb.sum()

# 描绘按月消费金额图
plt.plot(monthDf['实收金额'])
plt.title('按月消费金额图')
plt.xlabel('月份')
plt.ylabel('实收金额')
# 保存图片
plt.savefig('./month.png')
# 显示图片
plt.show()
 
# 聚合统计各种药品的销售数量
medicine = groupDf[['商品名称','销售数量']]
bk = medicine.groupby('商品名称')[['销售数量']]
re_medicine = bk.sum()
 
# 对药品销售数量按降序排序
re_medicine = re_medicine.sort_values(by='销售数量',ascending=False)
re_medicine.head()
# 截取销售数量最多的十种药品
top_medicine = re_medicine.iloc[:10,:]

# 用条形图展示销售数量前十的药品
top_medicine.plot(kind='bar')
plt.title('药品销售前十情况')
plt.xlabel('药品种类')
plt.ylabel('销售数量')
plt.legend(loc=0)
# 保存图片
plt.savefig('./medicine.png')
 # 显示图片
plt.show()
