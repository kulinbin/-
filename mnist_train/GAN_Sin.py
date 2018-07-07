# -*-coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
learning_rate = 0.0001
BATCH_SIZE = 64
SAMPLE_GAP = 0.2  # 采样间隔
SAMPLE_NUM = 50   # 采样次数
N_GNET = 10      # 噪声分布维度
# POINT = np.vstack([np.linspace(0, SAMPLE_NUM * SAMPLE_GAP, SAMPLE_NUM) for _ in range(BATCH_SIZE)])

POINT = np.linspace(0, SAMPLE_NUM * SAMPLE_GAP, SAMPLE_NUM)
plt.plot(POINT, np.sin(POINT))
plt.show()

def inputs():
    with tf.variable_scope("input"):
        real_in = tf.placeholder(tf.float32, [None, SAMPLE_NUM], name="real_in")  # [batch, 50]
        g_in = tf.placeholder(tf.float32, [None, N_GNET], name="Noise_in")  # [batch, 10]
    return real_in, g_in

def loss(y_real, y_fake):
    # 判别网络误差，用于在固定生成网络时，优化判别网络，主要包含两个部分，对于真实数据
    # 和虚假数据的判别误差
    with tf.variable_scope("loss"):
        d_loss = -tf.reduce_mean(tf.log(y_real) + tf.log(1-y_fake))
        g_loss = tf.reduce_mean(tf.log(1-y_fake))
    return d_loss, g_loss

def train_optimizer(d_loss, g_loss):
    # 定义训练操作
    with tf.variable_scope("train"):
        train_d = tf.train.AdamOptimizer(learning_rate).minimize(
            d_loss, var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator"))
        train_g = tf.train.AdamOptimizer(learning_rate).minimize(
            g_loss, var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator"
            ))
    return train_d, train_g

def fc_layer(input, size_in, size_out, name="fc"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", shape=[size_in,size_out], initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable("b", shape=[size_out], initializer=tf.constant_initializer(0.1))
        return tf.matmul(input, w) + b

def generator(noise_in):
    with tf.variable_scope("Generator"):
        g_l1 = tf.nn.relu(fc_layer(noise_in, N_GNET, 128, "l1"))
        g_out = fc_layer(g_l1, 128, SAMPLE_NUM, "g_out")
        return g_out

def discriminator(d_in, reuse=False):
    with tf.variable_scope("Discriminator", reuse=reuse):
        d_l0 = tf.nn.relu(fc_layer(d_in, SAMPLE_NUM, 128, "d_l1"))
        # 判别网络输出的是属于真实数据的概率，所以使用sigmoid函数
        d_logits = tf.nn.sigmoid(fc_layer(d_l0, 128, 1, "d_out"))
        return d_logits

def main():
    real_in, fake_in = inputs()
    d_real = discriminator(real_in)
    g_out = generator(fake_in)  # 从噪声分布采样生成伪造数据
    d_fake = discriminator(g_out, reuse=True)  # 对伪造图片判别
    d_loss, g_loss = loss(y_real=d_real, y_fake=d_fake)
    train_d, train_g = train_optimizer(d_loss, g_loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        plt.ion() # 开启interactive mode，便于连续plot
        for step in range(50000):
            # 在Sin上加入一点随机噪声，构成不同的真实输入数据
            real_data = np.vstack([np.sin(POINT) + np.random.normal(0, 0.01, SAMPLE_NUM) for _ in range(BATCH_SIZE)])
            # 生成网络的噪声输入
            g_noise = np.random.randn(BATCH_SIZE, N_GNET)
            g_output, g_prob, d_prob, _, _ = sess.run([g_out, d_fake, d_real, train_d, train_g],
                                                     feed_dict={real_in: real_data, fake_in: g_noise})
            # 将结果通过matplot 绘制，每隔500步更新一次
            if step % 500 == 0:
                plt.cla()
                plt.plot(POINT, g_output[0], c='#4AD631', lw=2, label="generated line") # 生成网络生成的数据
                plt.plot(POINT, real_data[0], c='#74BCFF', lw=3, label="real sin") # 真实数据
                prob = (d_prob.mean()+1-g_prob.mean())/2.
                plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % (prob),
                         fontdict={'size': 15})
                plt.ylim(-2, 2)
                plt.draw(),plt.pause(0.2)
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()