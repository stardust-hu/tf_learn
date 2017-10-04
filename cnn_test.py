# -*- coding: utf-8 -*-
# Created by yhu on 2017/10/4.
# Describe:

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape, bias=0.1):
    initial = tf.constant(bias, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w, strides=1):
    return tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')


def max_pool_2x2(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def interface(features, labels):
    features = tf.reshape(features, [-1, 28, 28, 1])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(features, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return train_step, cross_entropy, y_conv, keep_prob, accuracy


def train(max_step, checkpoint_dir='./checkpoint/'):
    features = tf.placeholder(tf.float32, shape=[None, 784])
    labels = tf.placeholder(tf.float32, shape=[None, 10])

    train_step, cross_entropy, y_conv, keep_prob, accuracy = interface(features, labels)

    mnist = input_data.read_data_sets('data', one_hot=True)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            sess.run(tf.global_variables_initializer())
            start_step = 0
            print('start training from new state')

        for step in range(start_step, start_step + max_step):
            start_time = time.time()
            batch = mnist.train.next_batch(50)
            sess.run(train_step, feed_dict={features: batch[0], labels: batch[1], keep_prob: 0.5})
            if step % 50 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={features: batch[0],
                                                               labels: batch[1],
                                                               keep_prob: 1})
                train_loss = sess.run(cross_entropy, feed_dict={features: batch[0],
                                                                labels: batch[1],
                                                                keep_prob: 1})
                duration = time.time() - start_time
                print("step %d: training accuracy %g, loss is %g (%0.3f sec)" % (step, train_accuracy, train_loss, duration))
            if step % 1000 == 1:
                saver.save(sess, './checkpoint/model.ckpt', global_step=step)
                print('writing checkpoint at step %s' % step)


def test(checkpoint_dir='./checkpoint/'):
    features = tf.placeholder(tf.float32, shape=[None, 784])
    labels = tf.placeholder(tf.float32, shape=[None, 10])

    train_step, cross_entropy, y_conv, keep_prob, accuracy = interface(features, labels)

    mnist = input_data.read_data_sets('data', one_hot=True)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            raise Exception('no checkpoint find')

        print('test accuracy %g' % accuracy.eval(feed_dict={features: mnist.test.images,
                                                            labels: mnist.test.labels,
                                                            keep_prob: 1.0}))


if __name__ == '__main__':
    # train(5000)
    test()
