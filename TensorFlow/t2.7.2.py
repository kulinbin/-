import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
sess=tf.Session()
batch_size=20
x_vals=np.random.normal(1,0.1,100)
y_vals=np.repeat(10.,100)
x_data=tf.placeholder(shape=[None,1],dtype=tf.float32)
y_target=tf.placeholder(shape=[None,1],dtype=tf.float32)
A=tf.Variable(tf.random_normal(shape=[1,1]))
my_output=tf.matmul(x_data,A)
init=tf.initialize_all_variables()
sess.run(init)
loss=tf.reduce_mean(tf.square(my_output-y_target))
my_opt=tf.train.GradientDescentOptimizer(0.02)
train_step=my_opt.minimize(loss)
loss_batch=[]

for i in range(100):
    rand_index = np.random.choice(100,size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    if (i + 1) % 5 == 0:
        print('step #' + str(i + 1) + 'A=' + str(sess.run(A)))
        temp_loss=sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('loss =' + str(temp_loss))
        loss_batch.append(temp_loss)

loss_stochastic=[]
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [[x_vals[rand_index]]]

    rand_y = [[y_vals[rand_index]]]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    if (i + 1) % 5 == 0:
        print('step #' + str(i + 1) + 'A=' + str(sess.run(A)))
        temp_loss=sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        print('loss =' + str(temp_loss))
        loss_stochastic.append(temp_loss)

plt.plot(range(0,100,5),loss_stochastic,'b-',label='stochastic loss')
plt.plot(range(0,100,5),loss_batch,'r--',label='batch loss')
plt.legend(loc='upper right',prop={'size':11})
plt.show()