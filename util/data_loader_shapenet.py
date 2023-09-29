import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append('..')
from cext import octree_database
from cext import primitive_points_suffix_index
  
def _add_data_to_queue(points, test):
  queue = tf.FIFOQueue(capacity=100,
      dtypes=[tf.float32])
  enqueue_op = queue.enqueue([points])
  numberOfThreads = 1 if test else 50
  qr = tf.train.QueueRunner(queue, [enqueue_op]*numberOfThreads)
  tf.train.add_queue_runner(qr)
  points = queue.dequeue()
  return points

def read_and_decode(filename_queue, batch_size, n_points, test=False):
  reader = tf.TFRecordReader()
  keys, serialized_examples = reader.read_up_to(filename_queue, batch_size)
  feature = {  
        'data': tf.FixedLenFeature([], tf.string),  
        # 'filename': tf.FixedLenFeature([], tf.string),  
    }  
  features = tf.parse_example(serialized_examples, features=feature)
  points = features['data']
  points = tf.io.decode_raw(points, out_type=tf.float32)
  points = tf.reshape(points, shape=[-1, n_points, 3])
  return _add_data_to_queue(points, test)


def data_loader(dataset, batch_size, n_points=5000, test=False):
  with tf.name_scope('read_and_decode'):
    filename_queue = tf.train.string_input_producer([dataset])
    points = read_and_decode(filename_queue, batch_size, n_points, test)
    node_position = primitive_points_suffix_index(tf.reshape(tf.transpose(points, [0, 2, 1]), shape=[batch_size, -1]))
    # node_position = points
  return points, node_position