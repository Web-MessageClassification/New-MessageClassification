# coding=utf-8
# Create by Lingfeng Lin

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from time import clock
import numpy as np
import jieba
import time
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
#统计数据的行数
trainData_count = len(open(var.train_data_path,'rU').readlines())
testData_count = len(open(var.test_data_path,'rU').readlines())

time_1 = time.time()
# 载入训练集
labelmat, datamat = loadtrain(var.train_data_path, maxline=800000)

# 载入不带标签的测试集
testdataMat = loadtest(var.test_data_path, maxline=200000)

# 使用jieba库进行中文分词
datamat = cutdata(datamat)
testdataMat = cutdata(testdataMat)

# 将字符串标签转换为整型
labelmat = [-1 if int(x) == 0 else 1 for x in labelmat]
# 交叉验证的步数
corssvalid_k = 5
corssvalid_q = [0] * corssvalid_k
corssvalid_recall = [1] * corssvalid_k
F1 = [1] * corssvalid_k
spamMassage_rate = [1] * corssvalid_k

corssvalid_time_train = [0] * corssvalid_k
corssvalid_time_test = [0] * corssvalid_k


time_2 = time.time()
print 'Read data cost ', time_2 - time_1, ' second', '\n'

for it in range(corssvalid_k):
    # 生成交叉验证的训练集和测试集
    cut_datamat, cut_labelmat, validdatamat, validlabelmat = makecrossvaliddata(datamat, labelmat, it, corssvalid_k)

    # 计算每个词的出现频率tf
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(cut_datamat)

    # 计算tf-idf矩阵
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print 'The column of tfidf matrix:', X_train_tfidf.shape

    time_3 = time.time()
    # 实例一个perceptron分类器
    # TODO 自己实现
    from sklearn.linear_model import Perceptron
    clf = Perceptron(penalty='l2', n_iter=5, eta0=0.05, random_state=0)
    clf.fit(X_train_tfidf, cut_labelmat)
    time_4 = time.time()
    corssvalid_time_train[it] = time_4 - time_3
    print 'Train cost ', time_4 - time_3, ' second', '\n'

    # 计算新数据tfidf
    X_new_counts = count_vect.transform(validdatamat)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    # 判别新数据
    predicted = clf.predict(X_new_tfidf)

    #计算测试数据的tfidf
    X_test_counts = count_vect.transform(testdataMat)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    time_5 = time.time()
    #判别测试数据的标签
    predicted_test = clf.predict(X_test_tfidf)
    time_6 = time.time()
    corssvalid_time_test[it] = time_6 - time_5
    print 'Test cost ', time_6 - time_5, ' second', '\n'

    #计算test数据的垃圾短信率
    spamMassage_rate[it] = np.mean(predicted_test == 1)

    #for i in range(len(predicted)):
    #   print('%d\t%s' % (predicted[i], validdatamat[i]))

    # 计算正确率, 召回率和F1-Measure
    pre_accurate = 0.0
    recall_accurate = 0.0
    sum_acc = 0.0
    sum_re = 0.0
    for i in range(len(predicted)):
        tmp_predit = int(predicted[i])
        tmp_valid = int(validlabelmat[i])
        # print('%d\t%s' % (predicted[i], validdatamat[i]))
        if tmp_predit == 1 and tmp_predit == tmp_valid:
            pre_accurate += 1
        if tmp_valid == 1 and tmp_valid == tmp_predit:
            recall_accurate += 1

        if tmp_predit == 1:
            sum_acc += 1
        if tmp_valid == 1:
            sum_re += 1
    # 计算正确率
    corssvalid_q[it] = pre_accurate / sum_acc
    # 计算召回率
    corssvalid_recall[it] = recall_accurate / sum_re
    # 计算F1-Measure
    F1[it] = 2.0 * corssvalid_q[it] * corssvalid_recall[it] / (corssvalid_q[it] + corssvalid_recall[it])

print 'Precision Rate: ', np.mean(corssvalid_q)
print 'Recall Rate:    ', np.mean(corssvalid_recall)
print 'F1-Measure:     ', np.mean(F1)
print 'SpamMassage Rate:', np.mean(spamMassage_rate)
print 'Average time of train:', np.mean(corssvalid_time_train)
print 'Average time of test:', np.mean(corssvalid_time_test)
finish = clock()
print '所用时间：'+str(finish - start)

#网格搜索得到的最优参数为{'penalty': 'l2', 'n_iter': 5, 'eta0': 0.05}
#Average time of train: 1.33980002403
#Average time of test: 0.0170000076294