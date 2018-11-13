#用tensorflow实现lasso回归和岭回归算法
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess=tf.Session()
iris=datasets.load_iris()
x_vals=np.array([x[3] for x in iris.data])
y_vals=np.array([y[0] for y in iris.data])
batch_size=50
learning_rate=0.001
x_data=tf.placeholder(shape=[None,1],dtype=tf.float32)
y_target=tf.placeholder(shape=[None,1],dtype=tf.float32)
A=tf.Variable(tf.random_normal(shape=[1,1]))
b=tf.Variable(tf.random_normal(shape=[1,1]))
model_output=tf.add(tf.matmul(x_data,A),b)

#增加损失函数，其为改良连续阶跃函数，lasso回归的截止点设为0.9 即限制斜率不超过0.9
lasso_param=tf.constant(0.9)
heavyside_step=tf.truediv(1.,tf.add(1.,tf.exp(tf.matmul(-100.,tf.subtract(A,lasso_param)))))
regularization_param=tf.matmul(heavyside_step,99.)
loss=tf.add(tf.reduce_mean(tf.square(y_target-model_output)),regularization_param)

# #岭回归
# ridge_param=tf.constant(1.)
# ridge_loss=tf.reduce_mean(tf.square(A))
# loss=tf.expand_dims(tf.add(tf.reduce_mean(tf.square(y_target-model_output)),tf.matmul(ridge_param,ridge_loss)),0)


#初始化变量 声明优化器
init=tf.global_variables_initializer()
sess.run(init)
my_opt=tf.train.GradientDescentOptimizer(learning_rate)
train_step=my_opt.minimize(loss)

#迭代
loss_vec=[]
for i in range(1500):
    rand_index=np.random.choice(len(x_vals),size=batch_size)
    rand_x=np.transpose([x_vals[rand_index]])
    rand_y=np.transpose([y_vals[rand_index]])
    sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
    temp_loss=sess.run(loss,feed_dict={x_data:rand_x,y_target:rand_y})
    loss_vec.append(temp_loss)
    if (i+1)%50==0:
        print('step#'+str(i+1)+' A='+str(sess.run(A))+' b='+str(sess.run(b)))
        print('loss='+str(temp_loss))
