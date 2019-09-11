#!/usr/bin/env python3
import numpy as np

A = np.array(  [[4.29, 2.2, 5.51],
                [5.20, 10.1,-8.24],
                [1.33, 4.8, -6.62]])
column_vectors = [A[::,i] for i in range(A.shape[1])]
column_v_norms = [np.linalg.norm(i) for i in column_vectors]
independent_columns = set([])

# using Cauchy-Schwatz to check independency. 
for i in range(len(column_vectors)):
    for j in range(len(column_vectors)):
        if i!=j:
            inner_prod = np.inner(column_vectors[i], column_vectors[j])
            if np.abs(inner_prod - column_v_norms[i]*column_v_norms[j]) == 0 \
                or np.abs(inner_prod - column_v_norms[i]*column_v_norms[j]) < 1E-5:
                independent_columns.add((min(i,j), max(i,j)))
print(independent_columns)