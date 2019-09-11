#!/usr/bin/env python3
import numpy as np

A = np.array(  [[0,1],
                [1,1],
                [1,1],
                [3,1],
                [3,1],
                [5,1]])
b = np.array([1,3.2,5,7.2,9.3,11.1])
A_t=np.transpose(A)
A_tA_1 = np.linalg.inv(A_t.dot(A))
q = (A_tA_1.dot(A_t)).dot(b)
print(q)