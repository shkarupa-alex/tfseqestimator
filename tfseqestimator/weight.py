from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.estimator.canned.head import _get_weights_and_check_match_logits

_FINAL_WEIGHTS_KEY = 'final_weights'


def sequence_weights_column():
    return tf.feature_column.numeric_column(_FINAL_WEIGHTS_KEY)


def make_sequence_weights(features, weight_column, sequence_logits, sequence_length):
    """Evaluate weight for sequence loss: [user_weight] * sequence_mask / sequence_length

    Args:
      features `dict` containing the input and sequence length information.
      weight_column A string or a `_NumericColumn` created by `tf.feature_column.numeric_column` defining
            feature column representing weights. It is used to down weight or boost examples during training.
      sequence_logits: `Tensor` shape `[batch_size, padded_length, ?]`.
      sequence_length: `Tensor` shape `[batch_size]`.

    Returns:
      `Tensor` with shape `[batch_size, padded_length]`.
    """

    # User provided weights
    user_weights = _get_weights_and_check_match_logits(features, weight_column, sequence_logits)
    user_weights = tf.convert_to_tensor(user_weights, dtype=tf.float32)

    # Per-item sequence mask
    batch_size, padded_length, _ = tf.unstack(tf.shape(sequence_logits))
    sequence_mask = tf.ones([batch_size, padded_length, 1])
    sequence_mask = _mask_real_sequence(sequence_mask, sequence_length)

    # Per-item length weights
    length_weights = tf.reshape(sequence_length, [batch_size, 1, 1])
    length_weights = tf.to_float(length_weights)

    # Final weights
    final_weights = tf.realdiv(sequence_mask, length_weights)
    final_weights = tf.multiply(user_weights, final_weights)
    final_weights = tf.squeeze(final_weights, axis=2)

    return {_FINAL_WEIGHTS_KEY: final_weights}


def _mask_real_sequence(sequence_input, sequence_length):
    """Set padded time steps to zero.

    Args:
      sequence_input: `Tensor` with shape `[batch_size, padded_length, ?]`.
      sequence_length: `Tensor` with shape `[batch_size]`.

    Returns:
      `Tensor` with same shape as sequence_input.
    """
    batch_size, padded_length, num_units = tf.unstack(tf.shape(sequence_input))

    # Create sequence mask
    output_mask = tf.sequence_mask(sequence_length, padded_length)
    output_mask = tf.expand_dims(output_mask, 2)
    output_mask = tf.tile(output_mask, [1, 1, num_units])

    # Mask padded outputs with zero
    sequence_output = tf.where(output_mask, sequence_input, tf.zeros_like(sequence_input, dtype=sequence_input.dtype))

    return sequence_output
