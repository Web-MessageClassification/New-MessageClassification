# coding=utf-8
# Create by lilelr

from numpy import *
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from time import clock

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


def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = trainMatrix.shape[0]
    numWords = trainMatrix.shape[1]
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = zeros(numWords); p1Num = zeros(numWords)      #change to ones()
    # p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones()
    p0Denom = 0.0; p1Denom = 0.0                        #change to 2.0

    # p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            # print trainMatrix[i]
            p1Denom += trainMatrix[i].sum()
        else:
            p0Num += trainMatrix[i]
            # print trainMatrix[i]
            p0Denom += trainMatrix[i].sum()
    p1Vect = p1Num/p1Denom          #change to log()
    p0Vect = p0Num/p0Denom         #change to log()
    # p1Vect = log(p1Num/p1Denom)          #change to log()
    # p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    # p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)

    p1 = sum(vec2Classify * p1Vec) *pClass1  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) *(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
    
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

if __name__ == '__main__':
    # 载入训练集
    labelmat, datamat = loadtrain('train.txt', maxline=800000)

    # 载入测试集
    testdataMat_raw = loadtest('test.txt', maxline=200000)

    # 使用jieba库进行中文分词
    datamat = cutdata(datamat)
    testdataMat = cutdata(testdataMat_raw)

    # 将字符串标签转换为整型
    labelmat = [0 if int(x) == 0 else 1 for x in labelmat]

    # 计算训练集中垃圾短信的比例
    # trashmsgs = 0
    # for i in range(len(labelmat)):
    #     if labelmat[i] == 1:
    #         trashmsgs += 1
    # print trashmsgs
    # print "原始数据垃圾短信的比例：" + str(double(trashmsgs) / double(len(labelmat)))

    word_count_vect = CountVectorizer()
    X_train_counts = word_count_vect.fit_transform(datamat)

    # 训练数据计算tf-idf矩阵，并训练出Logistc_regression模型
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # 训练出回归系数
    start = clock()
    p0V, p1V, pAb = trainNB0(X_train_tfidf, labelmat)
    # print p0V,p1V,pAb
    # print p0V.shape
    # print p1V.shape

    finish = clock()
    print '训练出模型所耗时间（秒）：' + str((finish - start))
    # print len(res_weights)

    # 计算测试数据tfidf
    X_test_counts = word_count_vect.transform(testdataMat)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    print X_test_tfidf.shape
    test_trashmsgs = 0.0
    # 测试数据结果写入
    test_answer = open('test_answer.txt', 'w')
    for i in range(X_test_tfidf.shape[0]):
        istrash = classifyNB(X_test_tfidf[i], p0V.transpose(), p1V.transpose(), pAb)
        test_answer.write(str(istrash) + '\t' + str(testdataMat_raw[i]))
        if istrash == 1:
            test_trashmsgs += 1.0

    test_answer.close()
    test_trash_ratio = test_trashmsgs / X_test_tfidf.shape[0]
    print '测试数据垃圾短信比例' + str(test_trash_ratio)