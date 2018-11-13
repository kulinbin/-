import tensorflow as tf
sess=tf.Session()
a=tf.Variable(tf.constant(4.))
x_vals=5.
x_data=tf.placeholder(dtype=tf.float32)
multiplication=tf.multiply(a,x_data)
loss=tf.square(tf.subtract(multiplication,50.))
init=tf.initialize_all_variables()
sess.run(init)
my_opt=tf.train.GradientDescentOptimizer(0.01)
train_step=my_opt.minimize(loss)

for i in range(10):
    sess.run(train_step,feed_dict={x_data:x_vals})
    a_val=sess.run(a)
    mult_output=sess.run(multiplication,feed_dict={x_data:x_vals})
    print(str(a_val)+'*'+str(x_vals)+'='+str(mult_output))

from tensorflow.python.framework import ops
ops.reset_default_graph()
sess=tf.Session()

a=tf.Variable(tf.constant(1.))
b=tf.Variable(tf.constant(1.))
x_vals=5.
x_data=tf.placeholder(dtype=tf.float32)
two_gate=tf.add(tf.multiply(a,x_data),b)
loss=tf.square(tf.subtract(two_gate,50.))
my_opt=tf.train.GradientDescentOptimizer(0.01)
train_step=my_opt.minimize(loss)
init=tf.initialize_all_variables()
sess.run(init)
print('\nOptimizing Two Gate Output to 50.')
for i in range(10):
    sess.run(train_step,feed_dict={x_data:x_vals})
    a_val,b_val=(sess.run(a),sess.run(b))
    two_gate_output=sess.run(two_gate,feed_dict={x_data:x_vals})
    print(str(a_val)+'*'+str(x_vals)+'+'+str(b_val)+'='+str(two_gate_output))