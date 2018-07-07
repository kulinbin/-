import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf
sess=tf.Session()

iris=datasets.load_iris()
binary_target