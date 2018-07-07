# -*-coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 64   # 一批数据大小
IMAGE_SIZE = 28     # 图片尺寸
N_GNET = 50        # 噪声分布大小
NUM_EPOCH = 10    # 数据集迭代批次

def conv2d(input, in_channel, out_channel, name="conv"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", shape=[5, 5, in_channel, out_channel],
                            initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable("b", shape=[out_channel], initializer=tf.constant_initializer(0.1))
        # 定义卷积步长为2，因此每次卷积长宽缩小2倍
        conv = tf.nn.conv2d(input, w, strides=[1,2,2,1], padding="SAME")
        return conv + b


def fc_layer(input, size_in, size_out, name="fc"):
    with tf.variable_scope(name):
        w = tf.get_variable("w", shape=[size_in,size_out], initializer=tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable("b", shape=[size_out], initializer=tf.constant_initializer(0.1))
        return tf.matmul(input, w) + b

# 定义lrelu激活函数
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x, name)

def disciminator(input, reuse=None):
    with tf.variable_scope("Discriminator", reuse=reuse):
        d_1 = lrelu(conv2d(input, in_channel=1, out_channel=32, name="d_1_conv"))
        d_2 = lrelu(conv2d(d_1, in_channel=32, out_channel=64, name="d_2_conv"))
        d_2_flatten = tf.reshape(d_2, shape=[-1, 7 * 7 * 64])
        d_3 = lrelu(fc_layer(d_2_flatten, 7 * 7 * 64, 512, name="d_3_lin"))
        return tf.nn.sigmoid(fc_layer(d_3, 512, 1), name="d_out")

def generator(input):
    with tf.variable_scope("Generator"):
        g_1 = tf.nn.relu(fc_layer(input, size_in=N_GNET, size_out=14*14, name="g_1_fc"))
        g_2 = tf.nn.sigmoid(fc_layer(g_1,size_in=14*14, size_out=28*28, name="g_2_fc"))
        g_3 = tf.reshape(g_2, shape=[BATCH_SIZE,28,28,1])
        return g_3

# 把batch_size个生成图片 拼接成size[0] x size[1]大小的整张图片，便于保存和可视化结果
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img

def inputs():
    with tf.variable_scope("input"):
        real_in = tf.placeholder(tf.float32, [None, 28*28*1], name="real_image")  # [batch, 784]
        g_in = tf.placeholder(tf.float32, [None, N_GNET], name="Noise_in")  # [batch, 50]
    return real_in, g_in

def loss(y_real, y_fake):
    with tf.variable_scope("loss"):
        d_loss = -tf.reduce_mean(tf.log(y_real) + tf.log(1-y_fake))
        g_loss = tf.reduce_mean(tf.log(1-y_fake))
    return d_loss, g_loss

def train_optimizer(d_loss, g_loss):
    with tf.variable_scope("train"):
        train_d = tf.train.AdamOptimizer(0.0001).minimize(
            d_loss, var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="Discriminator"))
        train_g = tf.train.AdamOptimizer(0.0002).minimize(
            g_loss, var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator"
            ))
    return train_d, train_g

def main():
    mnist = input_data.read_data_sets("data/mnist")  # 训练数据
    real_in, fake_in = inputs()  # 定义真实图片输入，生成网络噪声分布输入
    real_image = tf.reshape(real_in, [-1, 28, 28, 1])
    d_real = disciminator(real_image)  # 对真实图片判别
    fake_image = generator(fake_in)   # 从噪声分布采样生成伪造图片图片
    d_fake = disciminator(fake_image, reuse=True)  # 对伪造图片判别
    d_loss, g_loss = loss(y_real=d_real, y_fake=d_fake)  # 计算两个网络分别的误差
    train_d, train_g = train_optimizer(d_loss, g_loss)  # 训练优化器定义
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        current_epoch = 0
        step_per_epoch = mnist.train.num_examples // BATCH_SIZE
        plt.ion()
        while current_epoch < NUM_EPOCH:
            for step in range(step_per_epoch):
                real_data = mnist.train.images[step * BATCH_SIZE: (step+1) * BATCH_SIZE]
                g_noise = np.random.randn(BATCH_SIZE, N_GNET)
                g_output, d_prob, g_prob, d_losses, g_losses, _, _ = sess.run(
                    [fake_image, d_real, d_fake, d_loss, g_loss, train_d, train_g], feed_dict={real_in: real_data, fake_in: g_noise})
                if step % 200 == 0:
                    plt.clf()
                    print("Epoch:%d step:%d d_loss:%.2f, g_loss:%.2f" % (current_epoch, step,d_losses, g_losses))
                    plt.imshow(merge(g_output, [8, 8]))
                    plt.text(-10.0, -5.0, 'Epoch:%.2d step:%.4d D accuracy=%.2f (0.5 for D to converge)' %
                             (current_epoch, step, (d_prob.mean()+1-g_prob.mean())/2), fontdict={'size': 10})
                    plt.draw()
                    plt.pause(0.1)
            current_epoch += 1
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()