import numpy as np 
arr = np.array(([1, 2, 3, 4], [5,6, 7, 8], [9, 10, 11, 12] ))
print(f'arr:\n{arr}')
for column in range(arr.shape[1]):
    array_column_values = arr[:, column]
    print(f'array_column_values: {array_column_values}')
    print(f'mean: {np.mean(array_column_values)}')
