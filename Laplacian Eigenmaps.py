import numpy as np
from numpy import *
import matplotlib.pyplot as plt
def make_swiss_roll(n_samples=100, noise=0.0, random_state=None):
    
    #Generate a swiss roll dataset.

    t = 1.5 * np.pi * (1 + 2 * random.rand(1, n_samples))

    x = t * np.cos(t)

    y = 83 * random.rand(1, n_samples)

    z = t * np.sin(t)

    X = np.concatenate((x, y, z))

    X += noise * random.randn(3, n_samples)

    X = X.T #矩阵转置

    t = np.squeeze(t)

    return X, t


def laplaEigen(dataMat,k,t):
    #参数说明：k近邻=11,t为求边权值参数=5

    m,n=shape(dataMat) #m为点个数，n为维度

    W=mat(zeros([m,m])) #转为矩阵,类似于array函数

    D=mat(zeros([m,m])) #度矩阵

    for i in range(m):
        #当前节点dataMat[i,:]

        #k近邻操作
        k_index=knn(dataMat[i,:],dataMat,k)#选取前k个最近邻节点并返回索引

        for j in range(k):
            #第j个近邻节点dataMat[k_index[j],:]

            sqDiffVector = dataMat[i,:]-dataMat[k_index[j],:]

            sqDiffVector=array(sqDiffVector)**2

            sqDistances = sqDiffVector.sum() #距离计算

            #更新当前节点dataMat[i,:]与k邻近节点dataMat[k_index[j],:]边权重
            W[i,k_index[j]]=math.exp(-sqDistances/t) #权重矩阵，距离越近的两个节点边权重越大，没有连接的边权值为0

            D[i,i]+=W[i,k_index[j]] #节点度数

    L=D-W #拉普拉斯矩阵

    #np.linalg包含线性代数运算
    Dinv=np.linalg.inv(D) #矩阵求逆

    #matrix.I求逆
    X=np.dot(D.I,L) #矩阵乘法

    lamda,f=np.linalg.eig(X) #lamda为特征值，f为特征向量

    #返回特征值和特征向量
    return lamda,f

#knn算法构建图
def knn(inX, dataSet, k):
    
    # shape用于读取指定维度的长度
    dataSetSize = dataSet.shape[0]

    diffMat = tile(inX, (dataSetSize,1)) - dataSet

    sqDiffMat = array(diffMat)**2

    sqDistances = sqDiffMat.sum(axis=1)

    distances = sqDistances**0.5

    sortedDistIndicies = distances.argsort()    

    return sortedDistIndicies[0:k]

dataMat, color = make_swiss_roll(n_samples=2000) #生成数据集

print("shape:",shape(dataMat)," type:",type(dataMat)) #2000个三维数据点

lamda,f=laplaEigen(dataMat,11,5.0) #得到拉普拉斯矩阵的特征值和特征向量

fm,fn =shape(f) #特征向量矩阵，fm为特征向量维度，fn为特征向量个数

print ('fm,fn:',fm,fn)

lamdaIndicies = argsort(lamda) #对求得的特征值进行小到大排序，返回索引

first=0

second=0

print (lamdaIndicies[0], lamdaIndicies[1]) #前两个最小特征值的索引：0，1

for i in range(fm): #维度

    # .real为获取实部
    if lamda[lamdaIndicies[i]].real>1e-5:

        print (lamda[lamdaIndicies[i]])

        first=lamdaIndicies[i]

        second=lamdaIndicies[i+1]

        break

print (first, second)

redEigVects = f[:,lamdaIndicies]

fig=plt.figure('origin')

ax1 = fig.add_subplot(111, projection='3d')

#散点图
#参数说明：x集合、y集合和z集合
ax1.scatter(dataMat[:, 0], dataMat[:, 1], dataMat[:, 2], c=color,cmap=plt.cm.Spectral)


fig=plt.figure('lowdata')

ax2 = fig.add_subplot(111)

ax2.scatter(f[:,first].tolist(), f[:,second].tolist(), c=color, cmap=plt.cm.Spectral)

plt.show()

