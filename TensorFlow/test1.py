import numpy as np
import tensorflow as tf

x_vals=np.linspace(0,10,10)
# print(x_vals)

# y_vals=x_vals+np.random.normal(0,1,100)
x_vals_column=np.transpose(np.matrix(x_vals))
print(x_vals_column)
# x_vals_column1=np.matrix(x_vals)
#
# ones_column=np.transpose(np.matrix(np.repeat(1,100)))
#
# A=np.column_stack((x_vals_column,ones_column))
# print(A)