import csv

import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
from xgboost import XGBClassifier, plot_tree
import sklearn as sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree, preprocessing

# load data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from xgboost import plot_tree

dataset = pd.read_csv('data/Abalone Dataset.csv', delimiter=",")
dataset.head()
dataset.info()

# 构建训练集
X = dataset.iloc[:, 0:len(dataset.columns.tolist()) - 1].values
y = dataset.iloc[:, len(dataset.columns.tolist()) - 1].values
# split data into train and test sets
seed = 7
test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
# 标准化数据（可选）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# fit model no training data
# 训练
model = XGBClassifier()
model = model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("recall_score: %.2f%%" % recall_score(y_test, y_pred, average='micro'))
# print(sklearn.metrics.roc_auc_score(y_test, y_pred))
print("f1: %.2f%%" % sklearn.metrics.f1_score(y_test, y_pred, average='micro'))

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("ID3:")
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("recall_score:  %.2f%%" % recall_score(y_test, y_pred, average='micro'))
# print(sklearn.metrics.roc_auc_score(y_test, y_pred))
print("f1: %.2f%%" % sklearn.metrics.f1_score(y_test, y_pred, average='micro'))

with open("ID3.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f,
                             feature_names=['gender', 'length', 'diameter', 'height', 'totle_weight', 'shelling_weight',
                                            'visceral_weight', 'shell_weight'])

clf = tree.DecisionTreeClassifier(criterion='gini')
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("CART:")
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("recall_score:  %.2f%%" % recall_score(y_test, y_pred, average='micro'))
# print(sklearn.metrics.roc_auc_score(y_test, y_pred))
print("f1:  %.2f%%" % sklearn.metrics.f1_score(y_test, y_pred, average='micro'))

# Visualize model
with open("CART.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f,
                             feature_names=['gender', 'length', 'diameter', 'height', 'totle_weight', 'shelling_weight',
                                            'visceral_weight', 'shell_weight'])
