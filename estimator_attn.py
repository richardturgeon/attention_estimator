"""Binlingual TF Estimator model for CPU/GPU or Cerebras CS-1 """

import glob
import json
import logging
import os
import sys
import time

# Ours
from get_arguments import get_arguments
from tfrecord_data import input_fn
from hybrid_model  import model_fn

# Machine learning
#import numpy as np
import tensorflow as tf

# Cerebras
try:
    from cerebras.tf.cs_estimator import CerebrasEstimator as CommonEstimator
    from cerebras.tf.run_config import CSRunConfig as CommonRunConfig
    from cerebras.tf.cs_slurm_cluster_resolver import CSSlurmClusterResolver
    CEREBRAS_ENV = True
except:
    print("Cerebras support is not available")
    from common_classes import CommonEstimator
    from common_classes import CommonRunConfig
    CEREBRAS_ENV = False

from common_classes import validate_arguments

#                               # parameterize these 
DROPOUT = 0.20
SHUFFLE_BUFFER = 1500
NBR_CLASSES = 2

def logger(prefix):
    """ """
    logging.getLogger().setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)-2s - %(levelname)-2s - %(message)s', "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(prefix + 'attn_bin_estimator.log')
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)

    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(ch)
    return logging.getLogger(__name__)


def qualify_path(directory):
    """Generate fully qualified path name from input file name."""
    return os.path.abspath(directory)


def main(args):
    """ """

    # fname qualifier for inputs
    file_prefix = args.inpfx
    if file_prefix:
        file_prefix = file_prefix + '-'
    logger(file_prefix)

    epochs = args.epochs
    data_dir = qualify_path(args.data_dir)
    model_dir = qualify_path(args.model_dir)
    batch_size = args.batch_size

    metadata_file = os.path.join(data_dir, file_prefix + 'tfrecords-metadata.json')
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    eval_steps = metadata['val_samples'] // batch_size
    train_steps = metadata['train_samples'] // batch_size * epochs
    class_weights = metadata['class_weights']

    params = {}
    params['epochs'] = epochs
    params['data_dir'] = data_dir
    params['model_dir'] = model_dir
    params['batch_size'] = batch_size
    params['file_prefix'] = file_prefix
    params['train_steps'] = train_steps
    params['eval_steps'] = eval_steps
    params['class_weights'] = class_weights

    params['mode'] = args.mode
    params['learning_rate'] = args.learning_rate
    params['log_frequency'] = args.log_frequency

    params['shuffle_buffer'] = SHUFFLE_BUFFER
    params['nbr_classes'] = NBR_CLASSES
    params['dropout'] = DROPOUT
    params['input_sizes'] = (6212, 1)

    # Cerebras support
    params['cerebras'] = CEREBRAS_ENV
    params['cs_ip'] = args.cs_ip

    print("*" * 130)
    print(f"Batch size is {batch_size}")
    print(f"Number of epochs: {epochs}")
    print(f"Model directory: {model_dir}")
    print(f"Data directory: {data_dir}")
    print("params:", params)
    print("args:", args)
    print("*" * 130)

    if not validate_arguments(args.mode, CEREBRAS_ENV, params):
        print("Unable to continue, correct arguments or environment")
        return

    # Training may occur on the CPU or on the CS-1. CS-1 work is scheduled
    # via Slurm and requires posting configuration parameters to the 
    # TF-CONFIG environment variable.

    # should PORT_BASE be parameterized?
    # when should SEND_BLOCK and RECV_BLOCK workarounds be removed?

    if CEREBRAS_ENV and 'train' in args.mode:
        PORT_BASE = 23111
        slurm_cluster_resolver = CSSlurmClusterResolver(port_base=PORT_BASE)
        cluster_spec = slurm_cluster_resolver.cluster_spec()
        task_type, task_id = slurm_cluster_resolver.get_task_info()
        os.environ['TF_CONFIG'] = json.dumps({
            'cluster': cluster_spec.as_dict(),
            'task': {
                'type': task_type,
                'index': task_id
            }
        })

        #os.environ['SEND_BLOCK'] = '16384'
        #os.environ['RECV_BLOCK'] = '16384'

    # establish common Estimator and associated Config classes

    partition = None
    if 'predict' in args.mode:
        partition = 'test'
    if 'eval' in args.mode:
        partition = 'eval'
    if 'train' in args.mode:
        partition = 'train'
    params['partition'] = partition

    config = CommonRunConfig(
        cs_ip=params['cs_ip'],
        save_checkpoints_steps=train_steps,
        log_step_count_steps=1000
#       system_name="./02_fabric_connections_system14_sr.json",
    )

    model = CommonEstimator(
        use_cs=CEREBRAS_ENV,
        model_fn=model_fn,
        model_dir=model_dir,
        config=config,
        params=params,
    )

    # predict
    if 'predict' in args.mode:
        predict_not_impl = "PREDICT mode not yet implemented"
        tf.logging.error(predict_not_impl)
        assert False, predict_not_impl

    # train
    if 'train' in args.mode:
        print("\nTraining...", "on CS-1:", CEREBRAS_ENV)
        #train_input_fn = functools.partial(input_fn, data_dir, batch_size, partition='train', params=params)
        model.train(input_fn, steps=train_steps)
        print("Training complete")

    # evaluate 
    if 'eval' in args.mode:
        print("\nEvaluating...")
        #eval_input_fn = functools.partial(input_fn, data_dir, batch_size, partition='val', params=params)
        eval_result = model.evaluate(input_fn, steps=eval_steps)

        print("global step:%7d" % eval_result['global_step'])
        print("accuracy:   %7.2f" % round(eval_result['accuracy'] * 100.0, 2))
        print("loss:       %7.2f" % round(eval_result['loss'], 2))
        print("Evaluation complete")

    if 'compile_only' in args.mode or 'validate_only' in args.mode:
        print("\CS-1 preprocessing...")
        validate_only = 'validate_only' in args.mode
        #est_input_fn = functools.partial(input_fn, data_dir, batch_size, partition='val', params=params) # ??????
        model.compile(input_fn, validate_only=validate_only)
        print("\CS-1 preprocessing complete")

##______________________________________________________________________________
if __name__ == '__main__':
    arguments = get_arguments()
    main(arguments)
