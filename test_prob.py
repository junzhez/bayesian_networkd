from probability import *

b = ProbDist(['B'], {'B': [True, False]})

b[True] = 0.001
b[False] = 0.999

e = ProbDist(['E'], {'E': [True, False]})

e[True] = 0.002
e[False] = 0.998

a = ProbDist(['A', 'B', 'E'], {'A': [True, False], 'B': [True, False], 'E': [True, False]})

a[True, True, True] = 0.95
a[True, True, False] = 0.94
a[True, False, True] = 0.29
a[True, False, False] = 0.001
a[False, True, True] = 0.05
a[False, True, False] = 0.06
a[False, False, True] = 0.71
a[False, False, False] = 0.999

j = ProbDist(['A', 'J'], {'J' : [True, False], 'A' : [True, False]})
j[True, True] = 0.9
j[True, False] = 0.1
j[False, True] = 0.05
j[False, False] = 0.95

m = ProbDist(['A', 'M'], {'M' : [True, False], 'A' : [True, False]})
m[True, True] = 0.7
m[True, False] = 0.3
m[False, True] = 0.01
m[False, False] = 0.99

abe = a * (b * e)

ma = m
ja = j

m1 = ma.sum('M')
m2 = ja.sum('J')

m3 = (abe * m1).sum(['E', 'B'])
m4 = (abe * m2).sum(['E', 'B'])

b_ma = ma * m4
b_ja = ja * m3
b_abe = (abe * m1) * m2

def conditionalize(arr, dim, val):
    arr = arr.swapaxes(dim, 0)
    shape = arr.shape[1:]       # shape of the sub-array when we omit the desired dimension.
    count = np.array(shape).prod() # count of elements omitted the desired dimension.
    arr1 = arr.reshape(np.array(arr.shape).prod()) # flatten the array in-place.
    arr = arr1[val*count:(val+1)*count] # take the needed elements
    arr2 = arr1[(val+1)*count : (val +2)*count]
    aaa = [arr, arr2]
    print(aaa)
    arr = arr.reshape((1,)+shape) # the desired sub-array shape.
    arr = arr. swapaxes(0, dim)   # fix dimensions

    return arr
