# coding=utf-8
# Create by lilelr

from numpy import *
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from time import clock
import var


# 加载测试数据
def loadtrain(file_name, maxline=10):
    dataMat = []
    labelMat = []
    data = open(file_name).readlines()
    if maxline < len(data):
        data = data[:maxline]

    # msg_test = open('msgtest.txt','w')
    for line in data:
        # msg_test.write(str(line))
        lineArr = line.strip().split('\t')
        labelMat.append(lineArr[0])
        dataMat.append(lineArr[1])
    return labelMat, dataMat


# 加载测试数据
def loadtest(file_name, maxline=10):
    data = open(file_name).readlines()
    if maxline < len(data):
        data = data[:maxline]
    return data


# 分词
def cutdata(dataMat):
    return [' '.join(jieba.cut(line)) for line in dataMat]


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# sigmoid 数学函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

# 梯度上升法寻找最佳回归系数
def gradAscent(dataMatIn, classLabels):
    dataMatrix = dataMatIn
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001 # 向目标移动的步长
    maxCycles = 500 # 迭代次数
    weights = ones((n, 1)) # 所有回归系数初始化为1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # z=w0x0 + w1x1 + w2x2 + ... + wnxn
        error = (labelMat - h)  # 得到增长的向量
        weights = weights + alpha * dataMatrix.transpose() * error  # 修改下次迭代的回归系数
    return weights

# 随机梯度上升方法寻找最佳回归系数，与梯度上升法相比，一次仅用一个样本点更新回归系数，适用于数据量极大或特征成千上万的数据集
def stocGradAscent0(dataMatrix, classLabels):
    m, n = dataMatrix.shape[0],dataMatrix.shape[1]
    alpha = 0.01
    print n
    print m
    weights = ones(n)
    # print len(weights)
    for i in range(m):
        # # print dataMatrix[i].shape[1]
        # tmp_sum = 0.0
        # # for x in range(n):
        # #     tmp_sum += dataMatrix[i][x]*weights[x]
        # # print dataMatrix[i].shape
        # arr = array(dataMatrix[i])
        # # print arr
        # print arr
        # # for y in range(dataMatrix[i].shape[1]):
        # # print dataMatrix[i][y]
        # # tmp_ans = dataMatrix[i] * weights
        # # h = sigmoid(sum(dataMatrix[i] * weights))
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)  # initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001  # apha decreases with iteration, does not
            randIndex = int(random.uniform(0, len(dataIndex)))  # go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights


# 短信划分为正常和垃圾短信
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1
    else:
        return 0


# 训练数据交叉划分位训练集和测试集
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


if __name__ == '__main__':

    # 载入训练集
    labelmat, datamat = loadtrain('train.txt', maxline=1000)

    # 载入测试集
    testdataMat_raw = loadtest('test.txt', maxline=200)

    # 使用jieba库进行中文分词
    datamat = cutdata(datamat)
    testdataMat = cutdata(testdataMat_raw)

    # 将字符串标签转换为整型
    labelmat = [0 if int(x) == 0 else 1 for x in labelmat]

    # 计算训练集中垃圾短信的比例
    trashmsgs = 0
    for i in range(len(labelmat)):
        if labelmat[i] == 1:
            trashmsgs += 1
    print trashmsgs
    print "原始数据垃圾短信的比例：" + str(double(trashmsgs) / double(len(labelmat)))

    word_count_vect = CountVectorizer()
    X_train_counts = word_count_vect.fit_transform(datamat)

    # 训练数据计算tf-idf矩阵，并训练出Logistc_regression模型
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # 训练出回归系数
    start = clock()
    res_weights = gradAscent(X_train_tfidf, labelmat)
    finish = clock()
    print '训练出回归系数所耗时间（秒）：'+str((finish - start))
    print len(res_weights)

    # 计算测试数据tfidf
    X_test_counts = word_count_vect.transform(testdataMat)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    print X_test_tfidf.shape
    test_trashmsgs = 0.0
    # 测试数据结果写入
    test_answer = open('test_answer.txt', 'w')
    for i in range(X_test_tfidf.shape[0]):
        istrash = classifyVector(X_test_tfidf[i], res_weights)
        test_answer.write(str(istrash) + '\t' + str(testdataMat_raw[i]))
        if istrash == 1:
            test_trashmsgs += 1.0

    test_answer.close()
    test_trash_ratio = test_trashmsgs / X_test_tfidf.shape[0]
    print '测试数据垃圾短信比例' + str(test_trash_ratio)

    # 交叉验证的步数
    corssvalid_k = 5
    corssvalid_q = [0] * corssvalid_k
    corssvalid_recall = [1] * corssvalid_k
    F1 = [1] * corssvalid_k

    precision = 0.0
    start = clock()
    for it in range(corssvalid_k):
        # 生成交叉验证的训练集和测试集
        datamat, labelmat, validdatamat, validlabelmat = makecrossvaliddata(datamat, labelmat, it, corssvalid_k)

        # 计算每个词的出现频率tf
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(datamat)
        # print (X_train_counts.shape)
        # print (X_train_counts)

        # 计算tf-idf矩阵
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        # print (X_train_tfidf.shape)
        # print (X_train_tfidf)
        #
        # print ("he")
        # print (X_train_tfidf.size)
        # print (X_train_tfidf.ndim)
        # print "he"
        # dataArr,labeMat = loadDataSet()
        # 梯度上升方法计算回归参数
        res_weights = gradAscent(X_train_tfidf, labelmat)
        # 随机梯度上升方法
        # res_weights = stocGradAscent0(X_train_tfidf, labelmat)

        # print (res_weights)
        # print len(labelmat)

        # 计算新数据tfidf
        X_new_counts = count_vect.transform(validdatamat)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)

        errorcount = 0.0
        for i in range(X_new_tfidf.shape[0]):
            # print labelmat[i]%'，  '%classifyVector(X_train_tfidf[i],res_weights)
            # print str(labelmat[i])+","+str(classifyVector(X_train_tfidf[i],res_weights))
            niheans = classifyVector(X_new_tfidf[i], res_weights)
            # print int(niheans)
            if int(niheans) != int(validlabelmat[i]):
                errorcount += 1
        print errorcount
        total = X_new_tfidf.shape[0]
        tmp_precision = 1.0 - double(errorcount / total)
        precision += tmp_precision
        print (str(it) + "准确率：" + str(tmp_precision))
    finish = clock()
    print ('5交叉验证所用时间：'+str(finish - start))
    print ("总体准确率：" + str(precision / 5))
