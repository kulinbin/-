#-*-coding:utf-8-*-
import  matplotlib.pyplot as plt
import tensorflow as tf
sess=tf.Session()
x_vals=tf.linspace(-1.,1.,500)
target=tf.constant(0.)

#L2为欧拉损失函数 求目标值与预测值的平方和
l2_y_vals=tf.square(target-x_vals)
l2_y_out=sess.run(l2_y_vals)

#L1为绝对值损失函数
l1_y_vals=tf.abs(target-x_vals)
l1_y_out=sess.run(l1_y_vals)

#Pseudo-Huber损失函数，连续 平滑
delta1=tf.constant(0.25)
phuber1_y_vals=tf.multiply(tf.square(delta1),tf.sqrt(1.+tf.square((target-x_vals)/delta1))-1.)
phuber1_y_out=sess.run(phuber1_y_vals)

delta2=tf.constant(5.)
phuber2_y_vals=tf.multiply(tf.square(delta2),tf.sqrt(1.+tf.square((target-x_vals)/delta2))-1.)
phuber2_y_out=sess.run(phuber2_y_vals)


x_vals=tf.linspace(-3.,5.,500)
target=tf.constant(1.)
targets=tf.fill([500,],1.)
# #Hinge损失函数，主要用来评估支持向量机，有时也用于神经网络
hinge_y_vals=tf.maximum(0.,1.-tf.multiply(target,x_vals))
hinge_y_out=sess.run(hinge_y_vals)

#两类交叉熵损失函数
xentropy_y_vals=-tf.multiply(target,tf.log(x_vals)-tf.multiply((1.-target),tf.log(1.-x_vals)))
xentropy_y_out=sess.run(xentropy_y_vals)

#sigmoid
xentropy_sigmoid_y_vals=tf.nn.sigmoid_cross_entropy_with_logits(labels=x_vals,logits=targets)
xentropy_sigmoid_y_out=sess.run(xentropy_sigmoid_y_vals)

#test2.5.3
x_array=sess.run(x_vals)

plt.plot(x_array,l2_y_out,'b-',label='l2 loss')
plt.plot(x_array,l1_y_out,'r--',label='l1 loss')

plt.show()