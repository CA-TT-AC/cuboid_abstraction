import sys  
import tensorflow as tf  
  
sys.path.append('..')  
from cext import octree_conv  
from cext import octree_pooling  
  
  
def encoder(data, is_training=True, reuse=None):  
    with tf.variable_scope('encoder', reuse=reuse):  
        # Input transformation network  
        with tf.variable_scope('input_transform'):  
            data.set_shape([None, None, 3])  
            input_transform = tf.layers.conv1d(data, 64, kernel_size=1,  
                                               strides=1, activation=tf.nn.relu,  
                                               use_bias=True,  
                                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),  
                                               bias_initializer=tf.zeros_initializer(),  
                                               trainable=is_training)  
            input_transform = tf.layers.conv1d(input_transform, 128, kernel_size=1,  
                                               strides=1, activation=tf.nn.relu,  
                                               use_bias=True,  
                                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),  
                                               bias_initializer=tf.zeros_initializer(),  
                                               trainable=is_training)  
  
        # Feature transformation network  
        with tf.variable_scope('feature_transform'):  
            feature_transform = tf.layers.conv1d(input_transform, 64, kernel_size=1,  
                                                 strides=1, activation=tf.nn.relu,  
                                                 use_bias=True,  
                                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),  
                                                 bias_initializer=tf.zeros_initializer(),  
                                                 trainable=is_training)  
            feature_transform = tf.layers.conv1d(feature_transform, 128, kernel_size=1,  
                                                 strides=1, activation=tf.nn.relu,  
                                                 use_bias=True,  
                                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),  
                                                 bias_initializer=tf.zeros_initializer(),  
                                                 trainable=is_training)  
  
        # PointNet layers  
        with tf.variable_scope('pointnet_layers'):  
            pointnet = tf.layers.conv1d(feature_transform, 256, kernel_size=1,  
                                        strides=1, activation=tf.nn.relu,  
                                        use_bias=True,  
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),  
                                        bias_initializer=tf.zeros_initializer(),  
                                        trainable=is_training)  
            pointnet = tf.layers.conv1d(pointnet, 512, kernel_size=1,  
                                        strides=1, activation=tf.nn.relu,  
                                        use_bias=True,  
                                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),  
                                        bias_initializer=tf.zeros_initializer(),  
                                        trainable=is_training)  
  
        # Global max pooling  
        global_max_pooling = tf.reduce_max(pointnet, axis=1)  
  
        # Fully connected layers  
        with tf.variable_scope('fully_connected_layers'):  
            fc1 = tf.layers.dense(global_max_pooling, 256, activation=tf.nn.relu,  
                                  use_bias=True,  
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),  
                                  bias_initializer=tf.zeros_initializer(),  
                                  trainable=is_training)  
            fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu,  
                                  use_bias=True,  
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),  
                                  bias_initializer=tf.zeros_initializer(),  
                                  trainable=is_training)  
  
    return fc2  