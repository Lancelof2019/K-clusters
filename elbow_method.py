import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from matplotlib import pyplot as plt
from copy import deepcopy
iris=datasets.load_iris()
X=iris.data[:,2:4]

y=iris.target

k_list=list(range(1,17))

print(k_list)

sse_list=[0]*len(k_list)
print(sse_list)
enu_show=list(enumerate(k_list))
print(enu_show)
'''for k_ind,k in enumerate(k_list):
    print("-------------------------")
    print(k_ind)
    print(k)
    print("-------------------------")
'''

print("*************************************")


#sse_list=[]
for k_ind,k in enumerate(k_list):
    kmeans=KMeans(n_clusters=k,random_state=42)
    kmeans.fit(X)
    clusters_sk = kmeans.labels_
    #print("***************************")
    #print(clusters_sk)
    # print("***************************")
    centroids_sk = kmeans.cluster_centers_
    #print("***************************")
    #print(centroids_sk)
    #print("***************************")

    sse=0.0

    for i in range(k):
        cluster_i=np.where(clusters_sk==i)
        #print("cluster i is :")
        # print(cluster_i)
        #print("***********")
        #print(cluster_i.shape)
        #print("*******ssssss****")
        #print(X[cluster_i])
        #print("*******rrrrrr****")
        #print(centroids_sk[i])

        sse+=np.linalg.norm(X[cluster_i]-centroids_sk[i])
    print('k={}, SSE={}'.format(k,sse))
    #一次k=oo的循环求得的代价值
    sse_list[k_ind]=sse
plt.plot(k_list,sse_list)
plt.show()
