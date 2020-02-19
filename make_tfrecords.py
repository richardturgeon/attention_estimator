""" TFRecord generation """

import json
import logging
import numpy as np
import os
import pandas as pd
import sys
import sklearn
from   keras.utils import np_utils
from   sklearn.utils.class_weight import compute_class_weight
from   sklearn.model_selection import train_test_split
from   sklearn.preprocessing import StandardScaler
import tensorflow as tf

MYPATH = os.getcwd()
NBR_CLASSES = 2

def convert_to_tfr(x, y, path):
    """Write TFRecords from memory-resident features and label """

    print("writing to {}".format(path))

    with tf.io.TFRecordWriter(path) as writer:
        for i in range(x.shape[0]):
            # features
            x_curr = x[i]
            x_list = tf.train.FloatList(value=x_curr)
            x_feature = tf.train.Feature(float_list=x_list)

            # label
            y_curr = [y[i]]             # scalar value 
            y_list = tf.train.Int64List(value = y_curr)
            y_feature = tf.train.Feature(int64_list = y_list)
            feature_dict = {'data': x_feature, 'label': y_feature}

            # create the example and write a TFRecord
            feature_set = tf.train.Features(feature=feature_dict)
            example = tf.train.Example(features=feature_set)
            writer.write(example.SerializeToString())

            if i > 0 and i % 1000 == 0:
                sys.stdout.write("writing record {} \r".format(i))

        print("{} records written to {}".format(x.shape[0], path))

def main(csv_path=None, outpfx=None, test=False):
    """TFRecord generation."""

    # data originates in a CSV. The first row is a header containing column labels. 
    headers_df = pd.read_csv(csv_path, nrows=1).values
    nbr_cols = headers_df.size
    print("CSV header row size", nbr_cols)

    if not test:
        features_df = pd.read_csv(csv_path, skiprows=1).values.astype('float32')
    else:
        print("Generating TEST datasets")
        features_df = pd.read_csv(csv_path, nrows=10000, skiprows=1).values.astype('float32')
        if not outpfx:
            outpfx = 'TEST'

    if outpfx:
        outpfx += '-'

    y_df = features_df[:,0].astype('int')
    x_df = features_df[:, 1:nbr_cols]

    scaler = StandardScaler()
    x_df = scaler.fit_transform(x_df)

    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=.20, random_state=42)

    print('x_train shape:', x_train.shape)
    print('x_test_shape:', x_test.shape)

    weights = compute_class_weight('balanced', np.unique(y_train), y_train)

    y_train_negative, y_train_positive = np.bincount(y_train, minlength=2)  # positives are rare
    y_test_negative, y_test_positive = np.bincount(y_test, minlength=2)     # positives are rare

    y_train_total = y_train_negative + y_train_positive
    y_test_total = y_test_negative + y_test_positive
    negative = y_train_negative + y_test_negative
    positive = y_train_positive + y_test_positive
    total = y_train_total + y_test_total

    print('Examples: \n    Total: {}\n    Positive: {}\n {:.2f}% of total'.format(
        total, positive, 100 * positive / total))

    train_positive_pct = round((y_train_positive * 100 / y_train_total), 2)
    test_positive_pct = round((y_test_positive * 100 / y_test_total), 2)

    metadata = dict(
        train_samples=      int(y_train_total),
        train_positives=    int(y_train_positive),
        train_positive_pct= float(train_positive_pct),
        test_samples=       int(y_test_total),
        test_positives=     int(y_test_positive),
        test_positive_pct=  float(test_positive_pct),
        class_weights=      weights.tolist()
    )

    with open(outpfx + 'tfrecords-metadata.json', 'w') as f:
        json.dump(metadata, f)

    """
    # evaluate, use of scalar label fits better with existing Estimator implementations RRT 01/29/20
    y_train = np_utils.to_categorical(y_train, NBR_CLASSES)
    y_test = np_utils.to_categorical(y_test, NBR_CLASSES)
    """

    # write train and test tfrecord datasets
    train_name = outpfx + 'train.tfrecords'
    train_path = os.path.join(MYPATH, train_name)
    test_name = outpfx + 'test.tfrecords'
    test_path = os.path.join(MYPATH, test_name)

    convert_to_tfr(x_train, y_train, train_path)
    convert_to_tfr(x_test, y_test, test_path)

    print('TFRecords generated')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert CSV to TFRecords')

    parser.add_argument('--csv-path', type=str, help='CSV input file')
    parser.add_argument('--outpfx', type=str, default='', help='string prepended to output files')
    parser.add_argument('--test', default=False, action='store_true', help='generate small TFRecord datasets')
    args = vars(parser.parse_args())
    main(**args)
