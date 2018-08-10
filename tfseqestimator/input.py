from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
from tensorflow.contrib.feature_column import sequence_input_layer
from tensorflow.python.feature_column.feature_column import input_layer
from tensorflow.contrib.estimator.python.estimator.rnn import _concatenate_context_input


def build_sequence_input(features, sequence_columns, context_columns, input_partitioner,
                         sequence_dropout, context_dropout, is_training):
    """Combine sequence and context features into sequence input tensor.

    Args:
      features: `dict` containing the input `Tensor`s.
      sequence_columns: iterable containing `FeatureColumn`s that represent sequential model inputs.
      context_columns: iterable containing `FeatureColumn`s that represent model inputs not associated with a
        specific timestep.
      input_partitioner: partitioner for input layer variables.
      sequence_dropout: sequence input dropout rate, a number between [0, 1].
        When set to 0 or None, dropout is disabled.
      context_dropout: context input dropout rate, a number between [0, 1].
        When set to 0 or None, dropout is disabled.
      is_training: whether this operation will be used in training or inference.

    Returns:
      sequence_input: float `Tensor` of shape `[batch_size, padded_length, ?]`.
      _get_sequence_length: integer `Tensor` of shape `[batch_size]`.
    """

    with tf.variable_scope('input_features', values=tuple(six.itervalues(features)), partitioner=input_partitioner):

        sequence_input, sequence_length = sequence_input_layer(features=features, feature_columns=sequence_columns)
        tf.summary.histogram('sequence_length', sequence_length)

        if sequence_dropout:
            batch_size, padded_length, _ = tf.unstack(tf.shape(sequence_input))
            sequence_input = tf.layers.dropout(
                sequence_input,
                rate=sequence_dropout,
                noise_shape=[batch_size, padded_length, 1],
                training=is_training
            )

        if context_columns:
            context_input = input_layer(features=features, feature_columns=context_columns)

            if context_dropout:
                context_input = tf.layers.dropout(
                    context_input,
                    rate=context_dropout,
                    training=is_training
                )

            sequence_input = _concatenate_context_input(sequence_input, context_input)

        return sequence_input, sequence_length
