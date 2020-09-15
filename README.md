# K-clusters

if I change the [0]*len(X) to 0*len(X)

I get the error :

[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
0
[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16)]
*************************************
Traceback (most recent call last):
  File "C:/Workspace/Pycharm_projects/Elbow_methond.py", line 58, in <module>
    sse_list[k_ind]=sse
TypeError: 'int' object does not support item assignment
k=1, SSE=23.471159607768282
