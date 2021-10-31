import pandas as pd
import sklearn as sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree

# pandas 读取 csv 文件，header = None 表示不将首行作为列
data = pd.read_csv('data/spring.csv', header=None)
# 指定列
data.columns = ['season', 'after 8', 'wind', 'lay bed']

# sparse=False意思是不产生稀疏矩阵
vec = DictVectorizer(sparse=False)
# 先用 pandas 对每行生成字典，然后进行向量化
feature = data[['season', 'after 8', 'wind']]

X_train = vec.fit_transform(feature.to_dict(orient='record'))
# 打印各个变量
print('show feature\n', feature)
print('show vector\n', X_train)
print('show vector name\n', vec.get_feature_names())
print('show vector name\n', vec.vocabulary_)

Y_train = data['lay bed']
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, Y_train)

with open("out.dot", 'w') as f :
    f = tree.export_graphviz(clf, out_file = f,
            feature_names = vec.get_feature_names())