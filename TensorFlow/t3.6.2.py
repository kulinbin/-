#用tensorflow实现戴明回归算法，使得数据点距离回归直线之间的距离之和最短
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
sess=tf.Session()
iris=datasets.load_iris()
x_vals=np.array([x[3] for x in iris.data])
y_vals=np.array([y[0] for y in iris.data])
batch_size=50
x_data=tf.placeholder(shape=[None,1],dtype=tf.float32)
y_target=tf.placeholder(shape=[None,1],dtype=tf.float32)
A=tf.Variable(tf.random_normal(shape=[1,1]))
b=tf.Variable(tf.random_normal(shape=[1,1]))
model_output=tf.add(tf.matmul(x_data,A),b)

#损失函数 给定直线y=Ax+b,点(x0,y0) 两者之间的距离公式为 d=|y0-(Ax0+b)|/(根号A^2+1)
demming_numerator=tf.abs(tf.subtract(y_target,tf.add(tf.matmul(x_data,A),b)))       #公式分子
demming_denominator=tf.sqrt(tf.add(tf.square(A),1))                                 #公式分母
loss=tf.reduce_mean(tf.truediv(demming_numerator,demming_denominator))

#初始化变量，声明优化器，遍历迭代训练集以得到参数
init=tf.global_variables_initializer()
sess.run(init)
my_opt=tf.train.GradientDescentOptimizer(0.1)
train_step=my_opt.minimize(loss)

#迭代
loss_vec=[]
for i in range(2500):
    rand_index=np.random.choice(len(x_vals),size=batch_size)
    rand_x=np.transpose([x_vals[rand_index]])
    rand_y=np.transpose([y_vals[rand_index]])
    sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
    temp_loss=sess.run(loss,feed_dict={x_data:rand_x,y_target:rand_y})
    loss_vec.append(temp_loss)
    if (i+1)%50==0:
        print('step#'+str(i+1)+' A='+str(sess.run(A))+' b='+str(sess.run(b)))
        print('loss='+str(temp_loss))
[slope]=sess.run(A)
[y_intercept]=sess.run(b)
best_fit=[]
for i in x_vals:
    best_fit.append(slope*i+y_intercept)

plt.plot(x_vals,y_vals,'o',label='data')
plt.plot(x_vals,best_fit,'r-',label='best fit line',linewidth=3)
plt.legend(loc='upper left')
plt.xlabel('Pedal width')
plt.ylabel('sepal length')
plt.show()



