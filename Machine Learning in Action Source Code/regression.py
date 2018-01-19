# coding=utf-8
from numpy import *
import matplotlib.pyplot as plt


#生成测试数据
def generateData():
    x = [xi/100.0 for xi in range(1,100,1)]
    fileName = "ex0.txt"
    fw = open(fileName,'w')
    for xi in x:
        xii = 2.5+random.uniform(-0.1,0.1)
        fw.write(str(xii)+'\t')
        fw.write(str(xi)+'\t')
        fw.write(str(xii+2.0*xi)+'\n')
    fw.close()

#导入数据
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#标准回归函数
def standRegress(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

#画出标准回归函数原始数据散点图及预测曲线图
def standRegressPlot(xArr, yArr):
    ws = standRegress(xArr, yArr)
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat * ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])

    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat, 'r')
    plt.show()
    print xArr, yArr
    print ws

#局部加权线性回归函数（计算系数ws时为每个样本点引入权重）
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m)) #初始化样本点权重
    #根据样本点距离测试点的远近调整权重
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2)) #k控制衰减的速度
    xTx = xMat.T*(weights*xMat)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I*(xMat.T*(weights*yMat))
    return testPoint*ws

#测试局部加权线性回归函数
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

#画出局部加权线性函数原始数据散点图和拟合曲线
def lwlrPlot(xArr, yArr):
    xMat = mat(xArr)
    srtInd = xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]

    fig = plt.figure()
    #k=1.0
    yHat = lwlrTest(xArr, xArr, yArr, 1.0)
    ax = fig.add_subplot(311)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')

    #k=0.01
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    ax = fig.add_subplot(312)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')

    #k=0.003
    yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    ax = fig.add_subplot(313)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')

    plt.show()

#岭回归函数，给定lamda，返回回归系数
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx+eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = denom.I*(xMat.T*yMat)
    return ws

#岭回归函数测试，给定30个lamda，返回系数矩阵
def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat-yMean
    xMeans = mean(xMat,0)
    xVar = var(xMat,0)
    xMat = (xMat-xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:] = ws.T
    return wMat

#岭回归系数与lamda的关系图
def ridgePlot(xArr, yArr):
    ridgeWeights = ridgeTest(xArr,yArr)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

#将特征按照均值为0方差为1进行标准化处理
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat,0)
    inVar = var(inMat,0)
    inMat = (inMat-inMeans)/inVar
    return inMat

#均方误差
def rssError(yArr, yHatArr):
    return ((yArr-yHatArr)**2).sum()

#前向逐步线性回归（很暴力，初始系数为0，每次增加或减小一个数，用新系数计算估计值并得出误差，如果误差减小就作为新系数继续迭代）
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat-yMean
    xMat = regularize(xMat)
    m,n = shape(xMat)
    returnMat = zeros((numIt,n))
    ws = zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat

if __name__ == '__main__':
    generateData()
    xArr,yArr = loadDataSet('ex0.txt')
    #standRegressPlot(xArr,yArr)
    #lwlrPlot(xArr,yArr)
    #ridgePlot(xArr,yArr)
    stageWise(xArr,yArr,0.001,5000)

