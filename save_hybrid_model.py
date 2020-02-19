"""Construct a classification model_fn that can be used to instantiate
a CS-1 compatible tf.estimator.Estimator object for training.
"""

import tensorflow as tf
from keras_model import build_model
import estimator_hooks as hooks

def model_fn(features, labels, mode, params):
    """ """

    cerebras = params['cerebras']
    LR = params['learning_rate']

    # from Cerebras MNIST hybrid_model.py
    # tf.compat.v1.set_random_seed(0)       # --seed arg not yet implemented
    loss = None
    train_op = None
    logging_hook = None
    training_hook = None
    eval_metric_ops = None

    # living in the past?
    get_or_create_global_step_fn = tf.compat.v1.train.get_or_create_global_step
    get_global_step_fn = tf.compat.v1.train.get_global_step
    get_collection_fn = tf.compat.v1.get_collection
    set_verbosity_fn = tf.compat.v1.logging.set_verbosity
    optimizer_fn = tf.compat.v1.train.MomentumOptimizer
    accuracy_fn = tf.compat.v1.metrics.accuracy
    loss_fn = tf.compat.v1.losses.sparse_softmax_cross_entropy		# see loss_fn below
#   loss_fn = tf.nn.softmax_cross_entropy_with_logits_v2	        # see MNIST

    logging_INFO = tf.compat.v1.logging.INFO
    GraphKeys = tf.compat.v1.GraphKeys
    summary_scalar = tf.compat.v1.summary.scalar

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    is_evaluate = (mode == tf.estimator.ModeKeys.EVAL)
    is_predict  = (mode == tf.estimator.ModeKeys.PREDICT)

    inputs = features                       # no transformations needed
    keras_model = build_model(params, tensor=inputs)
    logits = keras_model.output
    predictions = tf.argmax(logits, 1)      # why axis=1?

    # label and class weights are concatenated, decouple
    weights = labels[:,1]
    float_labels = labels[:,0]
    labels = tf.cast(float_labels, dtype=tf.int64)                      # CS-1 flags "Cast"

    if is_training or is_evaluate:
        global_step = get_or_create_global_step_fn()
#       loss = loss_fn(labels=labels, logits=logits)                    # see loss_fn, weights= not supported in "v2"
#       loss = loss_fn(labels=labels, logits=logits, weights=weights)   # see loss_fn
#       squeezed_logits = tf.squeeze(logits)
#       loss = loss_fn(labels=labels, logits=squeezed_logits, weights=weights)  # see loss_fn
        loss = loss_fn(labels=labels, logits=logits, weights=weights)           # see loss_fn
        hook_list = []

        # hooks, metrics and scalars not available on Cerebras CS-1
        if not cerebras:
            accuracy = accuracy_fn(
                labels=labels,
                predictions=predictions,
                name='accuracy_op')

            eval_metric_ops = dict(accuracy=accuracy)
            summary_scalar('accuracy', accuracy[1])

            set_verbosity_fn(logging_INFO)
            logging_hook = tf.estimator.LoggingTensorHook(
                {"loss": loss, "accuracy": accuracy[1]},
                every_n_iter = 1000) #### every_n_secs = 60)

            hook_list.append(logging_hook)

        if is_training:
            optimizer = optimizer_fn(learning_rate=LR, momentum=0.9)
            update_ops = get_collection_fn(GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(
                    loss,
                    global_step=get_global_step_fn())

            training_hook = hooks.TrainingHook(params, loss)
            hook_list.append(training_hook)

        estimator = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            training_hooks=hook_list)

        return estimator

