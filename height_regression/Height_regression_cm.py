import openpyxl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split  #这里是引用了交叉验证


#获取数据
file_path = '身高预测参照表-1.xlsx'
data = openpyxl.load_workbook(file_path)
table = data.worksheets[0]
rows = table.rows
columns = table.columns
#将数据分割
regr = LinearRegression()
x = []
y = []
num = 0
for row in rows:
   if(num==0):
      num = 1
      continue
   temp = []
   temp.append(row[1].value)
   temp.append(row[2].value)
   y.append(row[3].value)
   x.append(temp)

# 获取训练集与测试集
X_train,X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)
regr.fit(X_train,y_train)

# 预测
y_pred = regr.predict(X_test)
print(len(y_test))
# 线性回归的测度，这里使用求均方根误差(Root Mean Squared Error, RMSE)的方式
import numpy as np
sum_mean=0
for i in range(len(y_pred)):
    sum_mean+=(y_pred[i]-y_test[i])**2
sum_erro=np.sqrt(sum_mean/len(y_pred))
# calculate RMSE by hand

# 获取得到的线性回归结果
num1 = regr.intercept_
num2 = regr.coef_[0]
num3 = regr.coef_[1]

print("结果为：身高 = "+str(num1)+" + "+str(num2)+"X1 + "+str(num3)+"X2.")
print("RMSE by hand:",sum_erro)
