import numpy as np

test = [
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [10,11,12],
    [13,14,15],
]


ind = 1
copy = list(test)
copy[ind] = [0,0,0]
print np.dot(test[ind],np.transpose(copy))
