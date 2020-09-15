import numpy as np
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


def visual_centroids(X,centroids,clusters):
    for i in range(k):
        cluster_i=np.where(clusters==i)
        plt.scatter(X[cluster_i,0],X[cluster_i,1])
    plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,c='#050505')
    plt.show()

#visual_centroids(X,centriods)



'''import numpy as np
from numpy import linalg as LA


a = np.array([-3, -5, -7, 2,  6,  4,  0,  2,  8])
b = a.reshape((3, 3))
print(b)
'''

'''


print(np.linalg.norm(b,axis=1))
print(np.linalg.norm(b,axis=0))
'''
def dist(a,b):
    a=np.array(a)
    b=np.array(b)
    divert=a-b
    #print(divert)
    #return np.linalg.norm(divert,axis=1)

    return np.linalg.norm(divert,axis=1)

#print(dist([2,4],[5,7]))

#print(np.linalg.norm(dist([2,4],[5,7]),axis=0))
#print(np.linalg.norm(dist([2,4],[5,7])))

def assigan_cluster(x,centriods):
    distances=dist(x,centriods)
    cluster=np.argmin(distances)
    return cluster

def update_centriods(X,centriods,clusters):
    for i in range(k):
        cluster_i=np.where(clusters==i)
        print(cluster_i)
        centriods[i]=np.mean(X[cluster_i],axis=0)
        print(centriods[i])


'''
random_test=np.random.randint(1,150)
print("random_test is : ", random_test)
random_vec=X[random_test]
print("The random vector is :")
print(random_vec)
print('--------------------')
print("The cluster of vector belongs to cluster_",assigan_cluster(random_vec,centriods))
'''
tol=0.0001
max_iter=100
iter=0
centriods_diff=100000
clusters=np.zeros(len(X))
print(clusters.shape)


while iter<max_iter and centriods_diff>tol:
      for i in range(len(X)):
          clusters[i]=assigan_cluster(X[i],centriods)
          print(" the item of the i:",i," data:",X[i],"belongs to cluster_",clusters[i])

      centroids_prev=deepcopy(centriods)
      update_centriods(X,centriods,clusters)
      iter+=1
      centriods_diff=np.linalg.norm(centriods-centroids_prev)
      print('Iterations: ',str(iter))
      print('centroids:\n',centriods)
      print('centriods moves: ,{:5.4f}'.format(centriods_diff))
      visual_centroids(X,centriods,clusters)
