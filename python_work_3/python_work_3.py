import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces


iris = load_iris()
x = iris.data
y = iris.target


def mean(X): #按列均值化
    return X-np.mean(X,axis = 0) 

def cov(X): #求协方差矩阵 公式： D*DT/m
    m = X.shape[0] #行
    cov_matrix = (1 / m) * X.T.dot(X) #这两种求解协方差矩阵都可以
    #cov_matrix = np.cov(X.T)       
    return cov_matrix

def feature_Value_Vector(X): #求解特征向量
    featureValue,featureVector = np.linalg.eig(X)
    featureValue = np.argsort(-featureValue) #按特征值从大到小排列
    return featureValue,featureVector

def subdim(featureValue,featureVector,mean):
    selectindex = featureValue[:3] #前3特征值
    selectbaseVector = featureVector[:,selectindex] #前3列
    processed_x = mean.dot(selectbaseVector) #原矩阵与选择前k列后的的矩阵相乘即为降维至k维的矩阵,dot 和 @ 一样作用
    #这里是selectbaseVector*mean 将mean映射到新选取的坐标轴上
    return processed_x


#============svd===============================================

def order(A):
    '''
    说明，这里是用来给特征值和特征矩阵排序的
    本质上是按每一列的第一行的元素（即特征值）的大小从大到小排序
    先转置即A.T 此时行变列，列变行
    然后按转置后的列（此时为特征值）从大到小排序。即np.argsort(-A.T[:,0])
    最后转回去.T
    '''
    return A.T[np.argsort(-A.T[:,0])].T


def svd(A):
    m,n = A.shape
    if m > n:
        x = np.linalg.eig(A.T @ A)
        #row_stack(x[0],x[1])合并特征值和特征向量矩阵
        X = order(np.row_stack((x[0].real,x[1].real)))
        sigma = np.sqrt(abs(X[0,]))
        V = X[1:,] #去掉特征值
        #np.diag是为了将奇异值形成一个对角线上的奇异值矩阵
        U = A @ V @ np.diag(1/sigma)
        return U,sigma,V
    else :
        U,S,V = svd(A.T)
        return V.T,S,U.T

if __name__=='__main__':
    plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] =False #负号显示

    Mean = mean(x) #均值化
    cov_matrix = cov(Mean) #协方差矩阵
    
    featureValue,featureVector = feature_Value_Vector(cov_matrix) #特征值、特征向量
    processed_x = subdim(featureValue,featureVector,Mean) #降维
    
    fig = plt.figure(figsize = (12,6))
    
    
    #画板1行2列2个图，第1个图，elev和azim为三维图的显示角度
    ax1 = fig.add_subplot(1,2,1,projection = "3d" ,elev = -150,azim =110)
    ax1.scatter(
        #x,y,z
        processed_x[:,0],processed_x[:,1],processed_x[:,2],
        c = y,
        cmap = plt.cm.Set1,
        edgecolor = "k",
        s = 40
        )
    ax1.set_title("My PCA 方向")
    ax1.set_xlabel("第一特征向量")
    ax1.w_xaxis.set_ticklabels([]) #去除坐标，坐标不对
    ax1.set_ylabel("第二特征向量")
    ax1.w_yaxis.set_ticklabels([])
    ax1.set_zlabel("第三特征向量")
    ax1.w_zaxis.set_ticklabels([])
    
    #降维为3                  
    #fit() 求得训练集X的均值，方差，最大值，最小值,这些训练集X固有的属性
    #transform() 在fit的基础上，进行标准化，降维，归一化等操作。
    #fit_transform(): fit和transform的组合，既包括了训练又包含了转换。
    X = PCA(n_components = 3).fit_transform(iris.data)
    ax2 = fig.add_subplot(1,2,2, projection="3d", elev=-150, azim=110)
    ax2.scatter(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        c=y,
        cmap=plt.cm.Set1,
        edgecolor="k",
        s=40,
    )
    ax2.set_title("sklearn.PCA 方向")
    ax2.set_xlabel("第一特征向量")
    ax2.w_xaxis.set_ticklabels([]) #去除坐标，坐标不对
    ax2.set_ylabel("第二特征向量")
    ax2.w_yaxis.set_ticklabels([])
    ax2.set_zlabel("第三特征向量")
    ax2.w_zaxis.set_ticklabels([])
    plt.show()


    #==================svd==================
    faceimages = fetch_olivetti_faces().images
    facelist = []
    for img in faceimages:
        facelist.append(img.flatten()) #从第0维展开，全转化为1维(400,64*64=4096)
    imgs = np.array(facelist)
    imgs_mean = np.mean(imgs,axis = 0) #求平均脸
    
    imgs-=imgs_mean #均值化

    '''
    所有脸
    sub = 1
    for image in imgs:
        t = image.reshape((64,64))#组装回2维
        plt.subplot(20,20,sub)
        plt.imshow(t,cmap = 'gray')
        plt.axis('off')
        sub+=1
    plt.show()
    '''
    U,sigma,V = svd(imgs)

    #从大到小前15个大的奇异值
    index = np.argsort(-sigma)[:15]
    topN = V[index,:]

    subs = 1
    plt.title("SVD")
    for image in topN:
        t = image.reshape((64,64))#组装回2维
        plt.subplot(3,5,subs)
        plt.imshow(t,cmap = 'gray')
        plt.axis('off')
        subs+=1
    plt.suptitle("SVD")
    plt.show()

    ave_image = imgs_mean.reshape(64,64)
    plt.imshow(ave_image,cmap = 'gray')
    plt.title("平均脸")
    plt.axis('off')
    plt.show()




