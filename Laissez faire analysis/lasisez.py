import csv

import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
from xgboost import XGBClassifier
import sklearn as sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree, preprocessing



# load data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('data/newLaissez.csv', delimiter=",")
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
print("recall_score: %f" % recall_score(y_test, y_pred))
print("AUC: %f" % sklearn.metrics.roc_auc_score(y_test, y_pred))
print("f1: %f" % sklearn.metrics.f1_score(y_test, y_pred))

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("recall_score: %f" % recall_score(y_test, y_pred))
print("AUC: %f" % sklearn.metrics.roc_auc_score(y_test, y_pred))
print("f1: %f" % sklearn.metrics.f1_score(y_test, y_pred))
with open("ID3.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f,
                             feature_names=['season', 'after 8', 'wind'])



clf = tree.DecisionTreeClassifier(criterion='gini')
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("recall_score: %f" % recall_score(y_test, y_pred))
print("AUC: %f" % sklearn.metrics.roc_auc_score(y_test, y_pred))
print("f1: %f" % sklearn.metrics.f1_score(y_test, y_pred))
print("clf: " + str(clf))

