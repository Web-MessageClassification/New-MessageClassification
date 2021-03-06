# coding=utf-8
# Create by lilelr

import numpy as np
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from time import clock
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

start = clock()
# 载入训练集
labelmat, datamat = loadtrain(var.train_data_path, maxline=10000)

# 载入测试集
# testdataMat = loadtest('test.txt', maxline=100)

# 使用jieba库进行中文分词
datamat = cutdata(datamat)
# testdataMat = cutdata(testdataMat)

# 将字符串标签转换为整型
labelmat = [0 if int(x) == 0 else 1 for x in labelmat]

#
# word_count_vect = CountVectorizer()
# # joblib.dump(word_count_vect,"word_count_vect.m")
# X_train_counts = word_count_vect.fit_transform(datamat)

# 训练数据计算tf-idf矩阵，并训练出Logistc_regression模型
# tfidf_transformer = TfidfTransformer()
# joblib.dump(tfidf_transformer,"tfidf_transformer")
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# # 训练出回归系数
# start = clock()
# clf = LogisticRegression()
# clf.fit(X_train_tfidf, labelmat)
# joblib.dump(clf,"lr_model.m")
# finsh = clock()
# print (finsh-start)

# 交叉验证的步数
corssvalid_k = 5
corssvalid_q = [0] * corssvalid_k
corssvalid_recall = [1] * corssvalid_k
F1 = [1] * corssvalid_k

for it in range(corssvalid_k):
    # 生成交叉验证的训练集和测试集
    cut_datamat, cut_labelmat, validdatamat, validlabelmat = makecrossvaliddata(datamat, labelmat, it, corssvalid_k)

    # 计算每个词的出现频率tf
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(cut_datamat)

    # 计算tf-idf矩阵
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print X_train_tfidf.shape


    # 实现Logistic Regression 分类器
    clf = LogisticRegression()
    clf.fit(X_train_tfidf, cut_labelmat)

    # 计算新数据tfidf
    X_new_counts = count_vect.transform(validdatamat)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    # 判别新数据
    predicted = clf.predict(X_new_tfidf)
    # 计算正确率, 召回率和F1-Measure
    pre_accurate =0.0
    recall_accurate = 0.0
    sum_acc = 0.0
    sum_re = 0.0
    for i in range(len(predicted)):
        tmp_predit = int(predicted[i])
        tmp_valid= int(validlabelmat[i])
        # print('%d\t%s' % (predicted[i], validdatamat[i]))
        if tmp_predit == 1 and tmp_predit==tmp_valid:
            pre_accurate+=1
        if tmp_valid == 1 and tmp_valid == tmp_predit:
            recall_accurate += 1

        if tmp_predit ==1:
            sum_acc+=1
        if tmp_valid == 1:
            sum_re+=1
    # 计算正确率
    corssvalid_q[it] = pre_accurate/sum_acc
    # 计算召回率
    corssvalid_recall[it] = recall_accurate/sum_re
    # 计算F1-Measure
    F1[it] = 2.0 * corssvalid_q[it] * corssvalid_recall[it] / (corssvalid_q[it] + corssvalid_recall[it])


print 'Precision Rate: ',np.mean(corssvalid_q)
print 'Recall Rate:    ',np.mean(corssvalid_recall)
print 'F1-Measure:     ',np.mean(F1)
finish = clock()
print '所用时间：'+str(finish - start)