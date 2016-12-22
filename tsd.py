import os
import random
import skimage.data
import skimage.transform
import numpy as np
import tensorflow as tf


def load_data(data_dir):
    """Loads a data set and returns two lists:

    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    images32 = [skimage.transform.resize(image, (32, 32))
                for image in images]
    data_images = np.array(images32)
    data_labels = np.array(labels)
    return data_images, data_labels

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_ys = v_ys
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    match_count = sum([int(y == y_) for y, y_ in zip(y_ys, y_pre)])
    result = match_count / len(y_ys)
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

train_data_dir = "/Users/Nathan/Documents/Study/Deep Learning/traffic-signs/traffic/traffic/datasets/BelgiumTS/Training"
test_data_dir = "/Users/Nathan/Documents/Study/Deep Learning/traffic-signs/traffic/traffic/datasets/BelgiumTS/Testing"
x, y = load_data(train_data_dir)
x_test, y_test = load_data(test_data_dir)

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 32, 32, 3])  # 32x32x3
ys = tf.placeholder(tf.int32, [None])
keep_prob = tf.placeholder(tf.float32)
# ys = tf.reshape(ys, [-1, 1])
x_image = tf.reshape(xs, [-1, 32, 32, 3])

## conv1 layer ##
W_conv1 = weight_variable([5, 5, 3, 32])  # patch 5x5, in size 3, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 32x32x32
h_pool1 = max_pool_2x2(h_conv1)  # output size 16x16x32


## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 16x16x64
h_pool2 = max_pool_2x2(h_conv2)  # output size 8x8x64

## fc1 layer ##
W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 8, 8, 64] ->> [n_samples, 8*8*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 62])
b_fc2 = bias_variable([62])
logits = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
prediction = tf.to_float(tf.argmax(logits, 1))

# the error between prediction and real data
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, ys))  # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


sess = tf.Session()
# important step
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sample_indexes = random.sample(range(len(x)), 100)
    sample_x = [x[j] for j in sample_indexes]
    sample_y = [y[j] for j in sample_indexes]
    sess.run(train_step, feed_dict={xs: sample_x, ys: sample_y, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(x_test, y_test))
