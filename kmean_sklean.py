import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from copy import deepcopy
iris=datasets.load_iris()

#print(iris)
X=iris.data[:,2:4]
print(X)
y=iris.target
print('--------------------------------------------------')
print(y)

#a={'shai':"Hello,laday",'shouer':"wuei",'no':1234}

#print(a['shai'])
#print(a['shouer'])

from matplotlib import pyplot as plt
y_0=np.where(y==0)
print("Y is :")
print(y_0)
plt.scatter(X[y_0,0],X[y_0,1])
y_1=np.where(y==1)
plt.scatter(X[y_1,0],X[y_1,1])
y_2=np.where(y==2)
plt.scatter(X[y_2,0],X[y_2,1])
'''
一堆数据在这里X[y_2,0]

'''
plt.show()

print(len(X))

print(np.size(X))
print(X.shape)
print(X.shape[0])
print(X.shape[1])
k=3
random_index=np.random.choice(range(len(X)),k)
print("The random number is :")
print(random_index)
print(random_index[0])
para1,para2,para3=random_index
centriods=X[random_index]
print(centriods)
print('--------------------')
print(X[para1,0],X[para1,1])
print(X[para2,0],X[para2,1])
print(X[para3,0],X[para3,1])

print(np.array([[X[para1,0],X[para1,1]],[X[para2,0],X[para2,1]],[X[para3,0],X[para3,1]]]))
print('--------------------')
kmeans_sk=KMeans(n_clusters=3,max_iter=400,random_state=42,tol=1e-7)

kmeans_sk.fit(X)

clusters_lab=kmeans_sk.labels_
print("The lable of cluster cluster_lab")
print(clusters_lab)
centroids_po=kmeans_sk.cluster_centers_

plt.scatter(X[:,0],X[:,1])
#clusters=np.zeros(len(X))
'''for i in range(k):
    clusters_i=np.where(clusters_lab==i)
    
    print("The cluster_",i," is \n",clusters_i)
    plt.scatter(X[clusters_i,0],X[clusters_i,1])
'''
plt.scatter(centroids_po[:,0],centroids_po[:,1],marker="*",s=200,c='#050505')
plt.show()
