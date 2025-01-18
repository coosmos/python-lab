import numpy as np

from array import array
arr1=np.array((1,5))
print(arr1)
arr2=array('i',[1,2,3,4,5])
print(arr2)
size=10000
li=range(size)
l2=range(size)

res=[]
for ele in range(size):
    res.append(li[ele]+l2[ele])
print(res)