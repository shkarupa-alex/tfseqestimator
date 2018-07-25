from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def make_sequence_weights(user_weights, sequence_logits, sequence_length):
    """Evaluate weight for sequence items loss: [user_weight] * sequence_mask / _get_sequence_length

    Args:
      user_weights: Scalar or `Tensor` with shape `[batch_size, padded_length, 1]` representing user-defined weights.
        It is used to down weight or boost examples during training.
      sequence_logits: `Tensor` shape `[batch_size, padded_length, ?]`.
      sequence_length: `Tensor` shape `[batch_size]`.

    Returns:
      `Tensor` with shape `[batch_size, padded_length, 1]`.
    """

    with tf.variable_scope('weights'):
        with tf.name_scope('user_weights'):
            # User provided weights
            user_weights = tf.convert_to_tensor(user_weights, dtype=tf.float32)

        with tf.name_scope('sequence_weights'):
            # Per-item sequence mask
            batch_size, padded_length, _ = tf.unstack(tf.shape(sequence_logits))
            sequence_weights = tf.ones([batch_size, padded_length, 1], dtype=tf.float32)
            sequence_weights = mask_real_sequence(sequence_weights, sequence_length)

        with tf.name_scope('final_weights'):
            # Final weights
            final_weights = tf.multiply(user_weights, sequence_weights)

        return final_weights


def mask_real_sequence(sequence_input, sequence_length):
    """Set padded time steps to zero.

    Args:
      sequence_input: `Tensor` with shape `[batch_size, padded_length, ?]`.
      sequence_length: `Tensor` with shape `[batch_size]`.

    Returns:
      `Tensor` with same shape as sequence_input.
    """

    with tf.variable_scope('sequence_mask'):
        batch_size, padded_length, num_units = tf.unstack(tf.shape(sequence_input))

        # Create sequence mask
        output_mask = tf.sequence_mask(sequence_length, padded_length)
        output_mask = tf.expand_dims(output_mask, 2)
        output_mask = tf.tile(output_mask, [1, 1, num_units])

        # Mask padded outputs with zero
        sequence_output = tf.where(output_mask, sequence_input,
                                   tf.zeros_like(sequence_input, dtype=sequence_input.dtype))

        return sequence_output
