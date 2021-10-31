# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 16:28:30 2021

@author: 云雨天阔
"""

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import plot_tree
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
import sklearn.svm
import sklearn.metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from graphviz import Source


dataset = pd.read_csv('data/iris.csv', delimiter=",")
dataset.head()
dataset.info()
print(dataset)

dataset.drop(dataset.columns[0], inplace=True, axis=1)
dataset.head()

print("total rows:",dataset.shape[0])
#查看两类标签的分类数量
print(dataset.Class.value_counts())

#构建训练集
X = dataset.iloc[:,0:len(dataset.columns.tolist())-1].values
y = dataset.iloc[:,len(dataset.columns.tolist())-1].values
# split data into train and test sets
seed = 7
test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
#标准化数据（可选）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# fit model no training data
#训练
model = XGBClassifier()
model=model.fit(X_train, y_train)


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
dtest_predprob = model.predict_proba(X_test)[:,1]


fig, ax = plt.subplots(figsize=(20,16))
plot_tree(model, num_trees=0, rankdir='LR',ax=ax)
plt.savefig('tree.jpg')
plt.show()

# plot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()

#对比逻辑回归算法
model_lg = LogisticRegression()
model_lg.fit(X_train, y_train)
y_pred = model_lg.predict(X_test)
print("Accuracy_score: %.2f%%" % (accuracy_score(y_test, y_pred)*100.0))
#print("Recall_score: %.2f%%" % (recall_score(y_test, y_pred)*100.0))
#print("Roc_auc_score: %.2f%%" % (sklearn.metrics.roc_auc_score(y_test, y_pred)*100.0))
#print("f1_score: %.2f%%" % sklearn.metrics.f1_score(y_test, y_pred))


#对比SVM算法


clf_svm = sklearn.svm.LinearSVC().fit(X_train, y_train)
decision_values = clf_svm.decision_function(X_train)
#precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_train, decision_values)
#plt.plot(precision,recall)
plt.legend(loc="lower right")
plt.show()
y_pred_svm = clf_svm.predict(X_test)
print(pd.crosstab(y_test, y_pred_svm, rownames=['Actual'], colnames=['Predicted']))
print(accuracy_score(y_test, y_pred_svm))
#print(recall_score(y_test, y_pred_svm))
#print(sklearn.metrics.roc_auc_score(y_test, y_pred_svm))
#print(sklearn.metrics.f1_score(y_test, y_pred_svm))