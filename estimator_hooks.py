import time
from datetime import datetime
import tensorflow as tf

class TrainingHook(tf.estimator.SessionRunHook):
    """Logs loss and runtime."""

    def __init__(self, params, tensor):
        super().__init__()
        self._tag = "***** Training Hook: "
        self.log_frequency = params['log_frequency']
        self.batch_size = params['batch_size']
        self.tensor = tensor

    def begin(self):
        self._step = -1
        self._start_time = time.time()
        print(self._tag, "begin()")

    def before_run(self, run_context):
        # print(self._tag, "before_run()")
        self._step += 1
        return tf.estimator.SessionRunArgs(self.tensor)

    def after_run(self, run_context, run_values):
        if self._step and self._step % self.log_frequency == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time
            loss_value = run_values.results
            examples_per_sec = self.log_frequency * self.batch_size / duration
            sec_per_batch = float(duration / self.log_frequency)

            format_str = (self._tag + '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print (format_str % (datetime.now(), self._step, loss_value,
                examples_per_sec, sec_per_batch))

