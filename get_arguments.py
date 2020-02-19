"""Acquire arguments """

import argparse

def get_arguments():
    parser = argparse.ArgumentParser("CS-1 evaluation std arguments")

    parser.add_argument('--model_dir',
                        default='./model_dir',
                        type=str,
                        help='tensorflow model_dir')

    parser.add_argument('--data_dir',
                        default='',
                        type=str,
                        help='tensorflow data_dir')

    parser.add_argument('--mode',
                        choices=[
                            'train',
                            'eval',
                            'predict',
                            'validate_only',        # CS-1 only
                            'compile_only'          # CS-1 only
                        ],
                        nargs='+',
                        required=True,
                        help='execution mode')

    parser.add_argument('--cs_ip',
                        default=None,
                        help='CS-1 IP address, default is None')

    parser.add_argument('--epochs',
                        default=1,
                        type=int,
                        help='Epochs')

    parser.add_argument('--batch_size',
                        default=32,
                        type=int,
                        help='Batch size')

    parser.add_argument('--learning_rate',
                        default=0.0001,         # investigate ?????????????
                        help='required in some configurations') # ?????????????

    parser.add_argument('--inpfx',
                        default='',
                        type=str,
                        help='prefix prepended to tfrecord, JSON and log files')

    parser.add_argument('--log_frequency',
                        default=1000,
                        type=int,
                        help='training run summarization interval')

    return parser.parse_args()

