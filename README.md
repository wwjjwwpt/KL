# KL
算法描述：  
降维：  
  第一步：归一化矩阵  
  第二步：计算协方差矩阵  
  第三步：利用协方差矩阵计算对角线矩阵. 
  第四步：排序对角线特征矩阵  
划分包腔  
  第一步：设定所需划分包腔数  
  第二步：利用k-means算法  
        随机初始化样本点的中心  
        对所有的样本点计算到每个初始化质心的距离  
        划分每个点对应最近的质心，并且重新计算质心  
        重复上数多次直到质心不变化  
核心程序:  
降维  
```
def kl(my_data, featureN):
    #求平均值
    my_data_mean = np.mean(my_data, axis=0)
    #矩阵X的平均值为0
    new_data = my_data - my_data_mean
    #计算协方差
    C = np.cov(new_data.T)
    #奇异值分解求使C对角得到矩阵
    c, p = np.linalg.eig(C)
    #特征值从大到小排序
    index = np.argsort(-c)
    c = c[index]
    p = p[:,index]
    P = p[:, 0:featureN]
    addmean =my_data_mean[:featureN]
    newmat = np.dot(new_data,P)+addmean
    return c, p, newmat
```  
kmeans
```
def kmeans(dataMat):
    # 训练模型
    km = KMeans(n_clusters=5)  # 初始化
    km.fit(dataMat)  # 拟合
    km_pred = km.predict(dataMat)  # 预测
    print(km_pred)
    centers = km.cluster_centers_  # 质心
    print(centers)

    # 可视化结果
    plt.scatter(np.array(dataMat)[:, 1], np.array(dataMat)[:, 0], c=km_pred)
    plt.scatter(centers[:, 1], centers[:, 0], c="r")
    plt.show()
import numpy as np

```
