# coding=utf-8
# Create by Lingfeng Lin

import numpy as np
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import var
import random
import time


class Perception(object):
    def __init__(self):
        self.learning_step = 0.00001
        self.max_iteration = 5000

    def loadtrain(self, file_name, maxline=10):
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


    def loadtest(self, file_name, maxline=10):
        data = open(file_name).readlines()
        if maxline < len(data):
            data = data[:maxline]
        return data


    def cutdata(self, dataMat):
        return [' '.join(jieba.cut(line)) for line in dataMat]


    def makecrossvaliddata(self, datamat, labelmat, it, k):
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




    def predict_(self, x):
        wx = sum([self.w[j] * x[j] for j in xrange(len(self.w))])
        return int(wx > 0)

    def train (self, X_train_tfidf, labelmat):

        X_train_tfidf = X_train_tfidf.todense()
        #选取初值w,b。
        self.w = [0.0] * (len(X_train_tfidf[0]) + 1)

        correct_count = 0
        time = 0

        while time < self.max_iteration:

            index = random.randint(0,len(labelmat)-1)
            x = list(X_train_tfidf[index])
            x.append(1.0)
            c = x
            y = 2 * labelmat[index] - 1
            wx = sum([self.w[j] * x[j] for j in xrange(len(self.w))])
            b = wx.sum()
            print b
            if b * y > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue

            for i in xrange(len(self.w)):
                self.w[i] += self.learning_step * (y * x[i])
                a = self.w[i]
                print i,self.w[i]
            print self.w
            print "interrupt"

    def predict(self,testset):
        labels = []
        for test in testset:
            x.list(test)
            x.append(1)
            labels.append(self.predict_(x))
        return labels


if __name__ == '__main__':
    print 'Start read data'

    time_1 = time.time()

    # 统计数据的行数
    trainData_count = len(open(var.train_data_path, 'rU').readlines())
    testData_count = len(open(var.test_data_path, 'rU').readlines())

    # 载入训练集
    p = Perception()
    labelmat, datamat = p.loadtrain(var.train_data_path, maxline=1000)

    # 载入不带标签的测试集
    testdataMat = p.loadtest(var.test_data_path, maxline=100)

    # 使用jieba库进行中文分词
    datamat = p.cutdata(datamat)
    testdataMat = p.cutdata(testdataMat)

    # 将字符串标签转换为整型
    labelmat = [-1 if int(x) == 0 else 1 for x in labelmat]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(datamat, labelmat, test_size=0.33,
                                                                                random_state=23323)
    # print train_features.shape
    # print train_features.shape

    # 计算每个词的出现频率tf
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_features)

    # 计算tf-idf矩阵
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # 计算带标签的测试数据的tf-idf矩阵
    X_new_counts = count_vect.transform(test_features)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    # 计算不带标签的测试数据的tf-idf矩阵
    X_test_counts = count_vect.transform(testdataMat)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)


    time_2 = time.time()
    print 'read data cost ', time_2 - time_1, ' second', '\n'

    print 'Start training'

    p.train(X_train_tfidf, train_labels)
    time_3 = time.time()
    print 'training cost ', time_3 - time_2, ' second', '\n'

    print 'Start predicting on label training data:'
    test_predict = p.predict(X_new_tfidf)
    time_4 = time.time()
    print 'predicting cost ', time_4 - time_3, ' second', '\n'

    score = accuracy_score(test_labels, test_predict)
    print "The accruacy socre on training data is ", score

    print 'Start predicting on no-label test data:'
    no_label_test_predict = p.predict(X_test_tfidf)
    time_5 = time.time()
    print 'predicting cost ', time_5 - time_4, ' second', '\n'

    score2 = accuracy_score(1, no_label_test_predict)
    print "The SpamMassage socre on no-label training data is ", score2











