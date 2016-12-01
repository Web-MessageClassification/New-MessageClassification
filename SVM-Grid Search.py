# coding=utf-8
# Created by FinsKap(Xiongziyi)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import jieba
import var

def loadtrain(file_name, maxline=10):
    dataMat = []
    labelMat = []
    data = open(file_name).readlines()
    if maxline < len(data):
        data = data[:maxline]
    for line in data:
        lineArr = line.strip().split('\t')
        labelMat.append(lineArr[0])
        dataMat.append(lineArr[1])
    return labelMat, dataMat


def loadtest(file_name, maxline=10):
    data = open(file_name).readlines()
    if maxline < len(data):
        data = data[:maxline]
    return data


def cutdata(dataMat):
    return [' '.join(jieba.cut(line)) for line in dataMat]


def makecrossvaliddata(datamat, labelmat, it, k):
    data = []
    label = []
    validdata = []
    validlabel = []
    for i in range(len(datamat)):
        if i % k == it:
            validdata.append(datamat[i])
            validlabel.append(labelmat[i])
        else:
            data.append(datamat[i])
            label.append(labelmat[i])
    return data, label, validdata, validlabel

# 载入训练集
labelmat, datamat = loadtrain(var.train_data_path, maxline=10000)

# 使用jieba库进行中文分词
datamat = cutdata(datamat)

# 将字符串标签转换为整型
labelmat = [-1 if int(x) == 0 else 1 for x in labelmat]

# 计算每个词的出现频率tf
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(datamat)

# 计算tf-idf矩阵
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# 设置网格参数（可自行增加）
Grid_parameters = [
{'kernel': ['rbf'], 'gamma': [0.1, 1, 10],'C': [1, 1.5, 2]},
{'kernel': ['linear'], 'C': [1, 1.5, 2]},
]

# 实例一个SVM分类器
SVM_Classifier = SVC(random_state=0)
clf = GridSearchCV(SVM_Classifier, Grid_parameters)
clf.fit(X_train_tfidf, labelmat)

# 输出最优参数
print(clf.best_params_)