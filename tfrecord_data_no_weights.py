import os
import tensorflow as tf

# ------------------------------------------------------------------------------
#   Estimator Input function (input_fn)
# ------------------------------------------------------------------------------

def input_fn(data_dir, batch_size, is_training=None, params=None):
    """Return dataset iterator."""

    epochs = params['epochs']
    file_prefix = params['file_prefix']
    shuffle_buffer = params['shuffle_buffer']
    data_sz, label_sz = params['input_sizes']

    if is_training:
        partition = 'train'
    else:
        partition = 'test'

    file_pattern = os.path.join(data_dir, f'{file_prefix}{partition}*.tfrecords')
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    def _parse_record_fn(raw_record):
        """Decode raw TFRecord into feature and label components."""

        feature_map = {
            'data':  tf.io.FixedLenFeature([data_sz], dtype=tf.float32),
            'label': tf.io.FixedLenFeature([1], dtype=tf.int64)
        }

        record_features = tf.io.parse_single_example(raw_record, feature_map)
        data = record_features['data']
        label = record_features['label']
        return data, label

    return process_record_dataset(dataset, is_training, batch_size, shuffle_buffer, _parse_record_fn, num_epochs=epochs)


def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer, parse_record_fn, num_epochs=None):
    """Given a Dataset with raw records, return an iterator over the records."""

    dataset = dataset.prefetch(buffer_size=batch_size)

    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
        dataset = dataset.repeat(count=None)    # forever, a CS-1 rqmt

    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda raw_record: parse_record_fn(raw_record),
            batch_size=batch_size,
#           num_parallel_batches=1,
#           num_parallel_calls=tf.contrib.data.AUTOTUNE,        # lots of warnings re inability to improve ... 
            drop_remainder=True))

    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset

