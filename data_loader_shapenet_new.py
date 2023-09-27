import tensorflow as tf

def parse_tfrecord_fn(example_proto):
    # 定义用于解析 TFRecord 中数据的函数
    feature_description = {
        'points': tf.FixedLenFeature([], tf.string),  # 注意这里的数据类型
    }
    example = tf.parse_single_example(example_proto, feature_description)
    
    # 解码二进制数据为张量
    points = tf.decode_raw(example['points'], tf.float32)
    
    # 这里可以对点云数据进行进一步的处理，例如重塑形状等
    
    return points

def data_loader(tfrecords_pattern, batch_size):
    # 从多个 TFRecords 文件中创建 dataset
    files = tf.train.match_filenames_once(tfrecords_pattern)
    filename_queue = tf.train.string_input_producer(files)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    # 解析 TFRecords 中的数据
    parsed_data = parse_tfrecord_fn(serialized_example)
    
    # 打乱和分批次数据
    data_batch = tf.train.shuffle_batch([parsed_data], batch_size=batch_size, capacity=10000, min_after_dequeue=5000)
    
    return data_batch