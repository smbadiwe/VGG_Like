"""A binary to train CIFAR-10 using a single GPU.
Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.
Speed: With batch_size 128.
System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def train(model_fn, train_folder):
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        # Get images and labels for CIFAR-10.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model_fn(images)

        # Calculate loss.
        loss = cifar10.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        global_step = tf.train.get_or_create_global_step()
        train_op = cifar10.train(loss, global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._start_time = time.time()

            def after_create_session(self, session, coord):
                self._step = session.run(global_step)

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        saver = tf.train.Saver(tf.global_variables())
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=train_folder,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                latest_checkpoint_path = tf.train.latest_checkpoint(train_folder)
                if latest_checkpoint_path:
                    # Restores from checkpoint
                    saver.restore(mon_sess, latest_checkpoint_path)
                    # # Assuming model_checkpoint_path looks something like:
                    # #   /my-favorite-path/cifar10_train/model.ckpt-0,
                    # # extract global_step from it.
                    # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                else:
                    print('No checkpoint file found')
                    return
                # # Restore the moving average version of the learned variables for eval.
                # variable_averages = tf.train.ExponentialMovingAverage(
                #     cifar10.MOVING_AVERAGE_DECAY)
                # variables_to_restore = variable_averages.variables_to_restore()
                # saver = tf.train.Saver(train_op)
                mon_sess.run(train_op)


def run_training(model_fn, qn_id):
    cifar10.maybe_download_and_extract()
    train_folder = FLAGS.train_dir + "_" + qn_id
    # if tf.gfile.Exists(train_folder):
    #     tf.gfile.DeleteRecursively(train_folder)
    # tf.gfile.MakeDirs(train_folder)
    train(model_fn, train_folder)
    print("Done running training for " + qn_id + "\n===================================\n")
    time.sleep(15)


def main(argv=None):  # pylint: disable=unused-argument
    run_training(cifar10.model_q1, "q1")
    run_training(cifar10.model_q2, "q2")
    run_training(cifar10.model_q3, "q3")


if __name__ == '__main__':
    tf.app.run()
