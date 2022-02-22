import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
from fancyimpute import KNN
from sklearn.linear_model import LassoCV, LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix, mean_absolute_error, \
    mean_squared_error, median_absolute_error, explained_variance_score, r2_score
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from apyori import apriori
import pyfpgrowth

# apriori算法
def apriori_method(data, min_support, min_confidence, min_lift, max_length):
    associate_rules = apriori(data, min_support=min_support,
                              min_confidence=min_confidence, min_lift=min_lift, max_length=max_length)

    for rule in associate_rules:
        print("频繁项集 %s，置信度 %f" % (rule.items, rule.support))
        for item in rule.ordered_statistics:
            print("%s -> %s, 置信度 %f 提升度 %f" %
                  (item.items_base, item.items_add, item.confidence, item.lift))
        print()

# FP-growth算法
def fpgrowth_method(data, min_support, min_confidence):
    # 频繁项集
    patterns = pyfpgrowth.find_frequent_patterns(data, min_support)
    # 规则
    rules = pyfpgrowth.generate_association_rules(patterns, min_confidence)
    print(rules)
    for i in rules:
        print("%s -> %s 置信度 %f" % (i, rules[i][0], rules[i][1]))


if __name__ == "__main__":
    # 统计缺失值
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    data = pd.read_excel('慢性肾病数据.xlsx', header=0)
    # 统计缺失值数量
    missing = data.isnull().sum().reset_index().rename(columns={0: 'missNum'})
    # 计算缺失比例
    missing['missRate'] = missing['missNum'] / data.shape[0]
    # 按照缺失率排序显示
    miss_analy = missing[missing.missRate > 0].sort_values(by='missRate', ascending=False)
    print(miss_analy)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # Show Chinese label
    plt.rcParams['axes.unicode_minus'] = False  # These two lines need to be set manually
    fig = plt.figure(figsize=(18, 10))
    plt.bar(np.arange(miss_analy.shape[0]), list(miss_analy.missRate.values), align='center',
            color=['red', 'green', 'yellow', 'steelblue'])

    plt.title('Histogram of missing value of variables')
    plt.xlabel('variables names')
    plt.ylabel('missing rate')
    # 添加x轴标签，并旋转90度
    plt.xticks(np.arange(miss_analy.shape[0]), list(miss_analy['index']))
    pl.xticks(rotation=90)
    # 添加数值显示
    for x, y in enumerate(list(miss_analy.missRate.values)):
        plt.text(x, y + 0.12, '{:.2%}'.format(y), ha='center', rotation=90)
    plt.ylim([0, 1.2])
    plt.show()
    print(data)
    # 缺失值填充
    # 利用knn算法填充
    fill_knn = KNN(k=3).fit_transform(data)
    data = pd.DataFrame(fill_knn)
    print(data)

    fpath = r"慢性肾病数据.xlsx"
    Dataset = pd.read_excel(fpath)
    data.columns = Dataset.columns

    print("------------------")
    print(data)

    x = data.loc[:, "性别":"eGFR"]
    y1 = data.loc[:, "CKD分层"]
    y2 = data.loc[:, "CKD评级"]
    names = x.columns
    names = list(names)
    key = list(range(0, len(names)))
    names_dict = dict(zip(key, names))
    names_dicts = pd.DataFrame([names_dict])

    # 训练集与测试集划分
    x_train, x_test, y_train, y_test = train_test_split(x, y2, test_size=0.33, random_state=7)
    """
    max_depth:树的最大深度
    """
    # 用逻辑回归的稳定性选择方法实现对特征的筛选
    model = xgb.XGBRegressor(max_depth=6, learning_rate=0.12, n_estimators=90, min_child_weight=6,
                             objective="reg:gamma")
    model.fit(x_train, y_train)
    print(model.feature_importances_)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # Show Chinese label
    plt.rcParams['axes.unicode_minus'] = False  # These two lines need to be set manually

    plt.title("Xgboost Feature Importance")
    plt.xlabel("Feature")
    plt.ylabel("Feature Score")
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.show()

    # 决策树
    print("决策树   ID3算法------------------------------------------")
    print("开始训练模型....")
    dt_ID3_model = DecisionTreeClassifier(criterion="entropy")
    dt_ID3_model.fit(x_train, y_train)  # 使用训练集训练模型
    predict_results = dt_ID3_model.predict(x_test)  # 使用模型对测试集进行预测
    print("正确率:")
    print(accuracy_score(predict_results, y_test))
    fig, ax = plt.subplots(figsize=(20, 16))
    plot_tree(dt_ID3_model, ax=ax)
    plt.savefig('决策树-ID3算法.jpg', dpi=1080)
    plt.show()

    # print("决策树   CART算法------------------------------------------")
    # print("开始训练模型....")
    # dt_CART_model = DecisionTreeClassifier()  # 参数均置为默认状态
    # dt_CART_model.fit(x_train, y_train)  # 使用训练集训练模型
    # predict_results = dt_CART_model.predict(x_test)  # 使用模型对测试集进行预测
    # print("正确率:")
    # print(accuracy_score(predict_results, y_test))
    # fig, ax = plt.subplots(figsize=(20, 16))
    # plot_tree(dt_CART_model, ax=ax)
    # plt.savefig('决策树-CART算法.jpg', dpi=1080)
    # plt.show()

    # 神经网络
    print("神经网络 ----------------------")
    mlp = MLPClassifier()
    mlp.fit(x_train, y_train)
    print('训练集准确率=', mlp.score(x_train, y_train))
    print('测试集准确率=', mlp.score(x_test, y_test))
    print('训练集混淆矩阵\n', confusion_matrix(y_train, mlp.predict(x_train)))
    print('测试集混淆矩阵\n', confusion_matrix(y_test, mlp.predict(x_test)))
    print('训练集分类报告\n', classification_report(y_train, mlp.predict(x_train), ))
    print('测试集分类报告\n', classification_report(y_test, mlp.predict(x_test)))

    #关联分析
    data1 = pd.read_excel('慢性肾病数据-关联分析.xlsx', header=0).values
    min_support = 0.1  # 最小支持度
    min_confidence = 0.5  # 最小置信度
    min_lift = 0.0  # 最小提升度
    max_length = 4  # 最长关系长度
    print('Apriori得到的关联规则')
    apriori_method(data1, min_support, min_confidence, min_lift, max_length)
    # print('FP-growth树得到的关联规则')
    # fpgrowth_method(data, min_support, min_confidence)

    # 线性回归
    lir = LinearRegression().fit(x_train, y_train)
    y_pred = lir.predict(x_test)
    ig = plt.figure(figsize=(10, 6))  ##设定空白画布，并制定大小
    plt.plot(range(y_test.shape[0]), y_test, color="blue", linewidth=1.5, linestyle="-")
    plt.plot(range(y_test.shape[0]), y_pred, color="red", linewidth=1.5, linestyle="-.")
    plt.legend(['真实值', '预测值'])
    plt.show()
    print('数据线性回归模型的平均绝对误差为：', mean_absolute_error(y_test, y_pred))
    print('数据线性回归模型的均方误差为：', mean_squared_error(y_test, y_pred))
    print('数据线性回归模型的中值绝对误差为：', median_absolute_error(y_test, y_pred))
    print('数据线性回归模型的可解释方差值为：', explained_variance_score(y_test, y_pred))
    print('数据线性回归模型的R方值为：', r2_score(y_test, y_pred))