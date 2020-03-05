from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class CNN:
    def __init__(self, train_data, train_label, val_data, val_label, val_data_dict, batch_size, epoch, lr, num_class):
        """
        Args: 
            train_data: Training data
            train_label: Traning label
            val_data: Validatoin data
            val_label: Validation label
            val_data_dict: Validation data per each slice
            batch_size: batch size for model training
            epoch: number of eopch for model training
            lr: learning rate
            num_class: Number of class
            
        """
        self.train_data = (train_data, train_label)
        self.val_data = (val_data, val_label)
        self.batch_size = batch_size
        self.epoch = epoch
        self.starter_learning_rate = lr
        self.val_data_dict = val_data_dict
        self.num_class = num_class

        self.x = tf.placeholder("float", shape=[None, 28, 28])
        self.y = tf.placeholder("float", shape=[None, 10])

        y_pred, logits = self.build_CNN_classifier(self.x)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=logits))

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.starter_learning_rate, global_step,150, 0.96, staircase=True)

        self.train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    
        
    def build_CNN_classifier(self, x):
        """ Builds a network """
        
        x_image = tf.reshape(x, [-1,28,28,1])
        conv1 = tf.layers.conv2d(inputs=x_image,
                                 filters = 4,
                                 kernel_size=[3, 3],
                                 padding="same",
                                 activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        conv2 = tf.layers.conv2d(inputs=pool1, 
                                 filters = 8, 
                                 kernel_size=[3, 3],
                                 padding="same", 
                                 activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        dim = 7 * 7 * 8
        X = tf.reshape(pool2, [-1, dim])
        W = tf.Variable(tf.truncated_normal(shape=[dim, 10], stddev=5e-2), name ="weight")
        b = tf.Variable(tf.constant(0.1, shape=[10]), name ="bias")
        
        logits = tf.matmul(X,W)+b
        y_pred = tf.nn.softmax(logits)
        return y_pred, logits

    def cnn_train(self):
        """ Trains the nework """
        
        def next_batch(num, data, labels):
            idx = np.arange(0 , len(data))
            np.random.shuffle(idx)
            idx = idx[:num]
            data_shuffle = [data[ i] for i in idx]
            labels_shuffle = [labels[ i] for i in idx]

            return np.asarray(data_shuffle), np.asarray(labels_shuffle)

        prev_loss = 100
        min_loss = 100
        loss_dict = dict()
        slice_num = self.check_num(self.train_data[1])
        
        gpu_options = tf.GPUOptions(visible_device_list="0")
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            for i in range(self.epoch):
                batch = next_batch(self.batch_size, self.train_data[0], self.train_data[1])
                sess.run(self.train_step, feed_dict={self.x: batch[0], self.y: batch[1]})
                if i % 10 == 0:
                    train_loss = self.loss.eval(feed_dict={self.x: batch[0], self.y: batch[1]})
                    val_loss = self.loss.eval(feed_dict={self.x: self.val_data[0], self.y: self.val_data[1]})
                    test_accuracy = self.accuracy.eval(feed_dict={self.x: self.val_data[0], self.y: self.val_data[1]})
#                     print("Epoch: %d, train_loss acc: %f, val_loss: %f, val_acc: %f" % (i, train_loss, val_loss, test_accuracy))
                
                if min_loss > val_loss:
                    min_loss = val_loss
                    for j in range(self.num_class):
                        loss_dict[j] = self.loss.eval(feed_dict={self.x: self.val_data_dict[j][0], self.y: self.val_data_dict[j][1]})
        
#         print(loss_dict, slice_num)
        return loss_dict, slice_num
                        
    def check_num(self, labels):
        """ Checks the number of data per each slice """
        
        slice_num = dict()
        for j in range(self.num_class):
            idx = np.argmax(labels, axis=1) == j
            slice_num[j] = len(labels[idx])
            
        return slice_num