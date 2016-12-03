# coding=utf-8
# Create by Dotomato(ChenJun)

import numpy as np
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
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
labelmat, datamat = loadtrain(var.train_data_path, maxline=800000)

# 载入测试集
# testdataMat = loadtest('test.txt', maxline=100)

# 使用jieba库进行中文分词
datamat = cutdata(datamat)
# testdataMat = cutdata(testdataMat)

# 将字符串标签转换为整型
labelmat = [-1 if int(x) == 0 else 1 for x in labelmat]

# 交叉验证的步数
corssvalid_k = 5
corssvalid_q = [0] * corssvalid_k
corssvalid_recall = [1] * corssvalid_k
F1 = [1] * corssvalid_k

for it in range(corssvalid_k):
    # 生成交叉验证的训练集和测试集
    datamat, labelmat, validdatamat, validlabelmat = makecrossvaliddata(datamat, labelmat, it, corssvalid_k)

    # 计算每个词的出现频率tf
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(datamat)

    # 计算tf-idf矩阵
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print X_train_tfidf.shape

    # 实例一个SVM分类器
    # TODO 自己实现
    clf = SGDClassifier(loss='hinge', alpha=1e-3, n_iter=5, random_state=42).fit(X_train_tfidf, labelmat)

    # 计算新数据tfidf
    X_new_counts = count_vect.transform(validdatamat)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    # 判别新数据
    predicted = clf.predict(X_new_tfidf)

    # for i in range(len(predicted)):
    #     print('%d\t%s' % (predicted[i], validdatamat[i]))

    # 计算正确率
    corssvalid_q[it] = np.mean(predicted == validlabelmat)
    # 计算正确率, 召回率和F1-Measure
    corssvalid_q[it] = np.mean(predicted == validlabelmat)
    corssvalid_recall[it] = 1.0 * len(validlabelmat) / len(labelmat) * np.mean(predicted == validlabelmat)
    F1[it] = 2.0 * corssvalid_q[it] * corssvalid_recall[it] / (corssvalid_q[it] + corssvalid_recall[it])

print 'Precision Rate: ',np.mean(corssvalid_q)
print 'Recall Rate:    ',np.mean(corssvalid_recall)
print 'F1-Measure:     ',np.mean(F1)
