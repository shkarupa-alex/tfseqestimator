from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import six
from tensorflow.contrib import layers as contrib_layers
from tensorflow.python.feature_column import feature_column


def build_sequence_input(sequence_columns, length_column, context_columns, features, params, is_training=False):
    """Combine sequence and context features into sequence input tensor.

    Args:
      features: `dict` containing the input and sequence length information.
      sequence_columns: iterable containing all the feature columns describing sequence (time steps) features.
        All items in the set should be instances of classes derived from `FeatureColumn`.
      length_column: features key or a `_NumericColumn`. Used as a key to fetch length tensor from features.
      context_columns: iterable containing all the feature columns describing context features i.e. features that
        apply across all time steps. All items in the set should be instances of classes derived from `FeatureColumn`.
      params: `HParams` instance with model parameters. Should contain:
          sequence_dropout: Sequence input dropout rate, a number between [0, 1].
            When set to 0 or None, dropout is disabled.
          context_dropout: Context input dropout rate, a number between [0, 1].
            When set to 0 or None, dropout is disabled.
      is_training: whether this operation will be used in training or inference.

    Returns:
      sequence_input: `Tensor` of dtype `float32` and shape `[batch_size, padded_length, ?]`.
      sequence_length: `Tensor` of dtype `int32` and shape `[batch_size]`.
    """

    # Transform input features
    columns_to_tensors = contrib_layers.transform_features(
        features,
        list(sequence_columns or []) + list(context_columns or [])
    )

    # Process sequence input
    sequence_input = contrib_layers.sequence_input_from_feature_columns(
        columns_to_tensors,
        sequence_columns,
    )

    batch_size, padded_length, _ = tf.unstack(tf.shape(sequence_input))

    if params.sequence_dropout:
        sequence_input = tf.layers.dropout(
            sequence_input,
            rate=params.sequence_dropout,
            noise_shape=[batch_size, padded_length, 1],
            training=is_training
        )

    if context_columns is not None:
        # Process context input
        context_input = contrib_layers.input_from_feature_columns(
            columns_to_tensors,
            context_columns,
        )
        if params.context_dropout:
            context_input = tf.layers.dropout(
                context_input,
                rate=params.context_dropout,
                training=is_training
            )

        # Combine sequence and context inputs
        sequence_input = _concatenate_context_input(sequence_input, context_input)

    # Get real sequence length
    sequence_length = _extract_sequence_length(features, length_column, batch_size, padded_length)

    return sequence_input, sequence_length


def _concatenate_context_input(sequence_input, context_input):
    """Replicate `context_input` across all timesteps of `sequence_input`.
    Expands dimension 1 of `context_input` then tiles it `sequence_length` times.
    This value is appended to `sequence_input` on dimension 2 and the result is returned.

    Args:
      sequence_input: `Tensor` of dtype `float32` and shape `[batch_size, padded_length, d0]`.
      context_input: `Tensor` of dtype `float32` and shape `[batch_size, d1]`.

    Returns:
      `Tensor` of dtype `float32` and shape `[batch_size, padded_length, d0 + d1]`.
    """
    seq_rank_check = tf.assert_rank(
        sequence_input,
        3,
        message='sequence_input must have rank 3',
        data=[tf.shape(sequence_input)]
    )
    seq_type_check = tf.assert_type(
        sequence_input,
        tf.float32,
        message='sequence_input must have dtype float32; got {}.'.format(sequence_input.dtype)
    )
    ctx_rank_check = tf.assert_rank(
        context_input,
        2,
        message='context_input must have rank 2',
        data=[tf.shape(context_input)]
    )
    ctx_type_check = tf.assert_type(
        context_input,
        tf.float32,
        message='context_input must have dtype float32; got {}.'.format(context_input.dtype)
    )
    with tf.control_dependencies([seq_rank_check, seq_type_check, ctx_rank_check, ctx_type_check]):
        padded_length = tf.shape(sequence_input)[1]
        tiled_context_input = tf.tile(
            tf.expand_dims(context_input, 1),
            tf.concat([[1], [padded_length], [1]], 0)
        )

    return tf.concat([sequence_input, tiled_context_input], 2)


def _extract_sequence_length(features, length_column, batch_size, padded_length):
    """Fetches sequence length from features and checks that it shape matches input.

    Args:
      features: `dict` containing the input and sequence length information.
      length_column: string or a `_NumericColumn` created by `tf.feature_column.numeric_column` defining
        feature column representing real sequence length.
        If it is a string, it is used as a key to fetch length tensor from the `features`.
        If it is a `_NumericColumn`, raw tensor is fetched by key `length_column.key`.
      batch_size: scalar size of input batch.
      padded_length: scalar maximum sequence length.

    Returns:
      `Tensor` of dtype `int32` and shape `[batch_size]`.
    """
    with tf.name_scope(None, 'length', values=tuple(six.itervalues(features)) + (batch_size, padded_length,)) as scope:
        default_length = tf.tile(tf.expand_dims(padded_length, 0), tf.expand_dims(batch_size, 0))
        if length_column is None:
            return default_length

        if isinstance(length_column, six.string_types):
            length_column = tf.feature_column.numeric_column(key=length_column, dtype=tf.int32)

        if not isinstance(length_column, feature_column._NumericColumn):
            error_msg = 'Length column must be either a string or _NumericColumn. Given type: {}.'
            raise TypeError(error_msg.format(type(length_column)))

        sequence_length = length_column._get_dense_tensor(feature_column._LazyBuilder(features))
        if not (sequence_length.dtype.is_floating or sequence_length.dtype.is_integer):
            error_msg = 'Length column should be castable to int. Given dtype: {}'
            raise ValueError(error_msg.format(sequence_length.dtype))

        sequence_length = tf.to_int32(sequence_length)
        sequence_length = tf.reshape(sequence_length, [-1])

        length_shape = tf.shape(sequence_length)
        assert_shape = tf.assert_equal(
            [batch_size],
            length_shape,
            message='length shape must be [batch_size]',
            data=[length_shape]
        )

        assert_value = tf.assert_greater_equal(
            default_length,
            sequence_length,
            message='length should be less or equal padded one',
            data=[sequence_length, default_length]
        )

        with tf.control_dependencies([assert_shape, assert_value]):
            return tf.identity(sequence_length, name=scope)
