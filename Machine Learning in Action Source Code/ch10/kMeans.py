# coding=utf-8
from numpy import *
import matplotlib.pyplot as plt

#辅助函数
#导入数据，创建数据集
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        #fltLine = map(float,curLine)
        fltLine = []
        fltLine.append(float(curLine[0]))
        fltLine.append(float(curLine[1]))
        dataMat.append(fltLine)
    return dataMat
#计算向量距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA-vecB,2)))
#根据给定数据集生成k个随机质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j])-minJ)
        centroids[:,j] = minJ+rangeJ*random.rand(k,1)
    return centroids

#K-均值聚类算法
def kMeans(dataSet, k, distMeas=distEclud,createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2))) #样本点分配结果
    centroids = createCent(dataSet,k) #保存质心
    clusterChanged = True #样本点所在簇是否改变
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
                clusterAssment[i,:] = minIndex,minDist**2
        print centroids
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust,axis=0)
    return centroids,clusterAssment

#二分K-均值聚类算法
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2))) #样本点簇分配结果，第一列簇下标，第二列距离质心的误差
    centroid0 = mean(dataSet,axis=0).tolist()[0] #初始质心
    centList = [centroid0] #所有的质心列表
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:])**2 #更新初始簇分配列表误差列
    while (len(centList) < k): #循环条件，质心数（簇数）小于指定数
        lowestSSE = inf #每次迭代假设初始总误差无穷大
        for i in range(len(centList)): #遍历质心（簇）进行拆分并计算误差
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:] #筛选出当前簇的样本点
            centroidMat,splitClustAss = kMeans(ptsInCurrCluster,2,distMeas) #对当前簇进行二元切分
            sseSplit = sum(splitClustAss[:,1]) #计算当前簇拆分后误差
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]) #计算非当前簇误差
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            if (sseSplit+sseNotSplit) < lowestSSE: #拆分簇+非拆分簇总误差小于初始总误差
                bestCentToSplit = i #记录最佳拆分簇
                bestNewCents = centroidMat #最佳拆分簇质心
                bestClustAss = splitClustAss.copy() #最佳拆分簇分配结果
                lowestSSE = sseSplit+sseNotSplit #更新总误差
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #更新最佳拆分簇分配结果簇下标
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit  # 更新最佳拆分簇分配结果簇下标
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ',len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:] #更新质心列表中拆分簇质心
        centList.append(bestNewCents[1,:]) #质心列表增加一个质心
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss \
            #将总分配结果中下标为拆分簇的部分更新为拆分簇的分配结果
    #原代码
    #return mat(centList), clusterAssment
    return row_stack(centList),clusterAssment

#用来显示图10-1和10-3
def showCluster(dataSet, k, centroids, clusterAssment):
    m, dim = shape(dataSet)
    if dim != 2:
        print ("Sorry! i can not draw because the dimension of data is not 2!")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print ("Sorry! Your k is too large!")
        return 1
    # draw all samples
    for i in range(m):
        markIndex = int(clusterAssment[i, 0])  # 为样本指定颜色
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], marker='+', color='red', markersize=18)
        # 用marker来指定质心样式，用color和markersize来指定颜色和大小

    plt.show()

if __name__ == '__main__':
    k = 4
    #datMat = mat(loadDataSet('testSet.txt'))
    #myCentroids,clustAssing = kMeans(datMat,k)
    #showCluster(datMat,k,myCentroids,clustAssing)
    #print myCentroids
    #print clustAssing

    datMat3 = mat(loadDataSet('testSet2.txt'))
    centList,myNewAssments = biKmeans(datMat3,k)
    showCluster(datMat3, k, centList, myNewAssments)
    print centList



