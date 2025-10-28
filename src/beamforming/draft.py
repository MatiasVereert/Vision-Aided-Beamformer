import numpy as np
from numpy.lib.stride_tricks import sliding_window_view# 1. Señal de entrada de un micrófono (longitud L=8)

m1 = np.array([10, 11, 12, 13, 14, 15, 16, 17])
m2 = np.array([0,  1,  2,  3,  4,  5,  6,  7])
m3 = np.array([20, 21, 22, 23, 24, 25, 26, 27])

array_signals = np.vstack((m1, m2, m3)) #shape (M,N)
K = 4

matrix_view = sliding_window_view(array_signals, K, axis = 1) #shape (M,N)
matrix_reverse = matrix_view[:,:,::-1]
print(matrix_reverse)

print(np.shape(matrix_reverse))