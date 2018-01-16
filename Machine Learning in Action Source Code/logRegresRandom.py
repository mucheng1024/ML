# coding=utf-8
import random

#用于逻辑回归生成数据集

def generateSet():
    fw = open('testSet.txt','w')
    for i in range(100):
        x = random.uniform(-3, 3)
        y = random.uniform(-5, 15)
        if x < 0 and y < 5:
            z = 1
        else:
            z = 0
        fw.write(str(x)+' ')
        fw.write(str(y)+' ')
        fw.write(str(z) + '\n')
    fw.close()

if __name__ == '__main__':
    generateSet()