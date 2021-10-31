#!/usr/bin/env python
# coding: utf-8

# In[6]:
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
#import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/release/bin/'
 



# load data
dataset = pd.read_csv('data/select-data.csv', delimiter=",")
dataset.head()
dataset.info()

with sns.plotting_context("notebook",font_scale=1.5):
    sns.set_style("whitegrid")
    sns.distplot(dataset["Age"].dropna(),
                 bins=20,
                 kde=False,
                 color="green")
    plt.ylabel("Count")
dataset["Geography"].value_counts().plot(x=None, y=None, kind='pie') 
boxplot1=sns.boxplot(x='Geography', y='Exited', data=dataset)
boxplot1.set(xlabel='Geography')
boxplot1=sns.boxplot(x='Gender', y='EstimatedSalary', data=dataset)
boxplot1.set(xlabel='Gender')
dataset.describe()
#total rows count
print("total rows:",dataset.shape[0])
#Detect null values
null_columns=dataset.columns[dataset.isnull().any()]
print(dataset[dataset.isnull().any(axis=1)][null_columns].count())

#去掉无用字段
dataset.drop(dataset.columns[0], inplace=True, axis=1)
dataset.head()
#查看两类标签的分类数量
dataset.Exited.value_counts()


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
print(accuracy_score(y_test, y_pred))

#对比SVM算法


clf_svm = sklearn.svm.LinearSVC().fit(X_train, y_train)
decision_values = clf_svm.decision_function(X_train)
precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_train, decision_values)
plt.plot(precision,recall)
plt.legend(loc="lower right")
plt.show()
y_pred_svm = clf_svm.predict(X_test)
print(pd.crosstab(y_test, y_pred_svm, rownames=['Actual'], colnames=['Predicted']))
print(accuracy_score(y_test, y_pred_svm))
print(recall_score(y_test, y_pred_svm))
print(sklearn.metrics.roc_auc_score(y_test, y_pred_svm))
print(sklearn.metrics.f1_score(y_test, y_pred_svm))


bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=3),
    n_estimators=500,
    learning_rate=.5,
    algorithm="SAMME")
bdt_discrete.fit(X_train, y_train)
discrete_test_errors = []

for discrete_train_predict in bdt_discrete.staged_predict(X_test):
    discrete_test_errors.append(1. - recall_score(discrete_train_predict, y_test))
n_trees_discrete = len(bdt_discrete)
# Boosting might terminate early, but the following arrays are always
# n_estimators long. We crop them to the actual number of trees here:
discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.plot(range(1, n_trees_discrete + 1),
         discrete_test_errors, c='black', label='SAMME')
plt.legend()
# plt.ylim(0.18, 0.62)
plt.ylabel('Test Error')
plt.xlabel('Number of Trees')
plt.subplot(132)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,
         "b", label='SAMME', alpha=.5)
plt.legend()
plt.ylabel('Error')
plt.xlabel('Number of Trees')
plt.ylim((.2,discrete_estimator_errors.max() * 1.2))
plt.xlim((-20, len(bdt_discrete) + 20))
plt.subplot(133)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights,
         "b", label='SAMME')
plt.legend()
plt.ylabel('Weight')
plt.xlabel('Number of Trees')
plt.ylim((0, discrete_estimator_weights.max() * 1.2))
plt.xlim((-20, n_trees_discrete + 20))
# prevent overlapping y-axis labels
plt.subplots_adjust(wspace=0.25)
plt.show()
y_pred_adaboost = bdt_discrete.predict(X_test)
print(accuracy_score(y_test, y_pred_adaboost))
print(recall_score(y_test, y_pred_adaboost))
# print(sklearn.metrics.roc_auc_score(y_test, y_pred))
print(sklearn.metrics.roc_auc_score(y_test, bdt_discrete.predict_proba(X_test)[:,1]))
print(sklearn.metrics.f1_score(y_test, y_pred_adaboost))

#随机森林
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 10, max_depth = 8, criterion = 'entropy',random_state = 42)
classifier_rf.fit(X_train, y_train)
# Predicting the Test set results
y_pred_rf = classifier_rf.predict(X_test)
print(pd.crosstab(y_test, y_pred_rf, rownames=['Actual Class'], colnames=['Predicted Class']))
print(accuracy_score(y_test, y_pred_rf))
print(recall_score(y_test, y_pred_rf))
print(f1_score(y_test, y_pred_rf))
print(sklearn.metrics.roc_auc_score(y_test, y_pred_rf))

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
fig, ax = plt.subplots(figsize=(10, 8))
probs_lg = model_lg.predict_proba(X_test)[:,1]
auc_lg = roc_auc_score(y_test, probs_lg)
print('Logististics AUC: %.3f' % auc_lg)
fpr_lg, tpr_lg, thresholds_lg = roc_curve(y_test, probs_lg)
probs_svm = y_pred_svm
auc_svm = roc_auc_score(y_test, probs_svm)
print('SVM AUC: %.3f' % auc_svm)
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, probs_svm)
probs_rf = y_pred_rf
auc_rf = roc_auc_score(y_test, probs_rf)
print('Random Forest AUC: %.3f' % auc_rf)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, probs_rf)
probs_xgb = dtest_predprob
# calculate AUC
auc_xgb = roc_auc_score(y_test, probs_xgb)
print('XGBoost AUC: %.3f' % auc_xgb)
# calculate roc curve
fpr_xgb, tpr_xgb, thresholds = roc_curve(y_test, probs_xgb)
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for models
pyplot.plot(fpr_xgb, tpr_xgb, marker='.',label='XgBoost')
pyplot.plot(fpr_lg, tpr_lg, marker='*',label='Logistics')
pyplot.plot(fpr_svm, tpr_svm, marker='o',label='SVM')
pyplot.plot(fpr_rf, tpr_rf, marker='^',label='RandomForest')
plt.ylabel('真正率')
plt.xlabel('假正率')
pyplot.legend(loc="best")
pyplot.show()




