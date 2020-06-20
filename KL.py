import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skimage.io as io
import matplotlib.pyplot as plt

# img = io.imread("flowers.jpg")
# print(type(img))
# io.imshow(img)
# plt.show()
my_data = np.loadtxt('ColorHistogram.cvs.txt').astype('float')
my_data = my_data[:, 1:]
print('数据集的大小为：',my_data.shape[0],my_data.shape[1])


# s = [[35, 21, 13], [1, 19, 33], [13, 38, 32], [2, 39, 8], [4, 9, 5], [41, 24, 24], [34, 22, 47], [15, 32, 17],
#       [47, 35, 29], [1, 37, 11]]
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




c,p,newmat = kl(my_data,2)
print('最大特征向量：',c.max())
kmeans(newmat)

