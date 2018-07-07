import tensorflow as tf
import numpy as np
sess=tf.Session()

# identity_matrix=tf.diag([1.0,1.0,1.0])
# A=tf.truncated_normal([2,3])
# B=tf.fill([2,3],5.0)
# C=tf.random_uniform([3,2])
# D=tf.convert_to_tensor(np.array([[1.1,2.,3.],[-3.,-7.,-1.],[0.,5.,-2.]]))
# print(identity_matrix)
# print(sess.run(identity_matrix))
# print(A)
# print(sess.run(A))
# print(B)
# print(sess.run(B))
# print(C)
# print(sess.run(C))
# print(D)
# print(sess.run(D))

print(sess.run(tf.cross([1,0],[0,1])))