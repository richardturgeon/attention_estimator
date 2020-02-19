"""TF Estimator and RunConfig wrappers

These tf.estimator.Estimator and tf.estimator.RunConfig class wrappers
allow Estimator-based scripts to run on either a traditional CPU/GPU, or
on a Cerebras CS-n accelerator without requiring source code changes. The
constructor functions of each class include the Cerebras specializations
implemented by new keyword arguments. Those arguments are simply swallowed.

Usage
-----

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

# The CEREBRAS_ENV variable can be used to tailor the remainder of the script.
"""

import logging
import tensorflow as tf

class CommonEstimator(tf.estimator.Estimator):

    def __init__(self, use_cs=None, **kwargs):
        super(CommonEstimator, self).__init__(**kwargs)

class CommonRunConfig(tf.estimator.RunConfig):

    def __init__(self, cs_ip=None, **kwargs):
        super(CommonRunConfig, self).__init__(**kwargs)


def validate_arguments(mode_list, is_cerebras, params_dict):
    """Estimator script argument/environment validation """
    if 'validate_only' in mode_list or 'compile_only' in mode_list:
        if not is_cerebras:
            tf.logging.error("validate_only and compile_only not available")
            return False

    if is_cerebras and 'train' in mode_list:
        if not params['cs_ip']:
            tf.logging.error("--cs_ip is required when training on the CS-1")
            return False

        if ':' not in params['cs_ip']:
            params['cs_ip'] += ':9000'              # why? 

    return True

