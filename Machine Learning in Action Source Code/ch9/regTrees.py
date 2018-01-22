# coding=utf-8
from numpy import *
import matplotlib.pyplot as plt

#生成测试数据
def generateData():
    fw = open('ex01.txt', 'w')
    for i in range(100):
        x = random.uniform(0, 1)
        y = random.uniform(-0.5, 1.5)
        fw.write(str(x) + '\t')
        fw.write(str(y) + '\n')
    fw.close()

#创建数据集
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    return dataMat

#二元切分，将数据分成两个子集
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    # 下面原书代码报错 index 0 is out of bounds,使用上面两行代码
    # mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
    # mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
    return mat0,mat1

#叶节点生成函数
def regLeaf(dataSet):
    return mean(dataSet[:,-1])
#误差计算函数
def regErr(dataSet):
    return var(dataSet[:,-1])*shape(dataSet)[0]

#回归树的切分函数，选择最佳切分点和切分值
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0)+errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S-bestS) < tolS:
        return None, leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex,bestValue

#递归创建树
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None: return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree

#二次切分的数据点图
def ex01Plot():
    myDat = loadDataSet('ex01.txt')
    myMat = mat(myDat)
    createTree(myMat)
    plt.plot(myMat[:,0],myMat[:,1],'r.')
    plt.show()

#多次切分的数据点图
def ex02Plot():
    myDat1 = loadDataSet('ex02.txt')
    myMat1 = mat(myDat1)
    createTree(myMat1)
    plt.plot(myMat1[:, 1], myMat1[:, 2],'r.')
    plt.show()

#回归树剪枝函数
#判断一个节点是树节点还是叶节点
def isTree(obj):
    return (type(obj).__name__=='dict')

#获取给定数的某两个相邻叶节点的平均值
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

#剪枝
def prune(tree, testData):
    if shape(testData)[0] == 0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + \
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge:
            print "merging"
            return treeMean
        else:
            return tree
    else:
        return tree

#测试剪枝
def testPrune():
    myDat2 = loadDataSet('ex2.txt')
    myMat2 = mat(myDat2)
    myTree = createTree(myMat2,ops=(0,1))
    #导入测试数据
    myDatTest = loadDataSet('ex2test.txt')
    myMat2Test = mat(myDatTest)
    print prune(myTree, myMat2Test)

#模型树，对于给定数据集拟合出线性回归函数，基于线性回归函数生成叶节点以及计算误差
#生成线性回归函数
def linearSolve(dataSet):
    m,n = shape(dataSet)
    X = mat(ones((m,n)))
    Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse, \n\
                        try increasing the second value of ops')
    ws = xTx.I*(X.T*Y)
    return ws,X,Y

#叶节点生成函数
def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

#误差计算函数
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X*ws
    return sum(power(Y-yHat,2))

def testModelTree():
    myMat2 = mat(loadDataSet('exp2.txt'))
    print createTree(myMat2,modelLeaf,modelErr,(1,10))
    plt.plot(myMat2[:,0],myMat2[:,1],'r.')
    plt.show()

if __name__ == '__main__':
    #generateData()
    #ex01Plot()
    #ex02Plot()
    #testPrune()
    testModelTree()


