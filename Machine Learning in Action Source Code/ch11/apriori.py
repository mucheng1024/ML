# coding=utf-8

#基本思路：
# 对于Ck，计算支持度保留频繁项集生成Lk；
# 根据Lk生成C(k+1)，计算支持度保留频繁项集生成L(k+1)；
# 以此类推，直到Ln中数量为0或1；
# 其中，根据Lk生成C(k+1)的过程：
# 合并Lk中前k-2个元素相同的项生成C(k+1)

#Apriori算法中的辅助函数
#创建数据集
def loadDataSet():
    return [[1,3,4], [2,3,5], [1,2,3,5], [2,5]]
#创建C1
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset,C1)
#计算支持度，过滤Ck生成Lk
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList,supportData

#Apriori算法
#根据Lk生成C(k+1)
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

#完整的apriori算法
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = map(set,dataSet)
    L1, supportData = scanD(D,C1,minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2],k)
        Lk, supK = scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L,supportData

#关联规则生成函数
#生成关联规则主函数
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet] #单元素的右端列表
            if (i > 1): #项集元素数量大于2
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf) \
                #递归生成关联规则
            else: #项集元素数目等于2
                calcConf(freqSet,H1,supportData,bigRuleList,minConf) \
                #根据给定项集和右端列表，计算可信度，保存关联规则，并返回筛选后的右端列表用于生成k+1的右端列表
    return bigRuleList

#根据给定项集和右端列表，生成关联规则计算可信度，筛选符合条件的规则和右端列表
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print freqSet-conseq,'-->',conseq,'conf: ',conf
            brl.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH

#根据给定项集和右端列表，合并右端列表生成k+1列表，调用calcConf方法筛选关联规则和右端列表，然后对项集和新列表进行递归调用
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m+1)):
        Hmp1 = aprioriGen(H,m+1)
        Hmp1 = calcConf(freqSet,Hmp1,supportData,brl,minConf)
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet,Hmp1,supportData,brl,minConf)

if __name__ == '__main__':
    dataSet = loadDataSet()
    C1 = createC1(dataSet)
    D = map(set,dataSet)
    L1, suppData0 = scanD(D,C1,0.5)
    L, suppData = apriori(dataSet,minSupport=0.5)
    rules = generateRules(L,suppData,minConf=0.5)
    print dataSet
    print C1
    print D
    print L1
    print L[1]
    print rules