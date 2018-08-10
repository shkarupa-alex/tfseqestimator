from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
from tensorflow.python.estimator.canned.dnn import _add_hidden_layer_summary


class DenseActivation(object):
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'

    @classmethod
    def all(cls):
        return cls.RELU, cls.SIGMOID, cls.TANH

    @classmethod
    def validate(cls, key):
        if callable(key):
            return
        elif isinstance(key, six.string_types) and key in cls.all():
            return
        else:
            raise ValueError('Invalid activation type {} or name: {}'.format(type(key), key))

    @classmethod
    def instance(cls, key):
        if callable(key):
            return key
        elif cls.RELU == key:
            return tf.nn.relu
        elif cls.SIGMOID == key:
            return tf.nn.sigmoid
        elif cls.TANH == key:
            return tf.nn.tanh
        else:
            return None


def build_input_dnn(sequence_input, dense_layers, dense_activation, dense_dropout, dense_norm, is_training):
    """Build a time-distributed dense network for input features.

    Args:
      sequence_input: `Tensor` with shape `[batch_size, padded_length, d0]`. Each input time step will be represented
        as separate example in dense layer.
      dense_layers: iterable of integer number of hidden units per layer. Only negative values will be used.
      dense_activation: activation function for dense layers. One of `DenseActivation` options or callable.
      dense_dropout: dropout rate, a number between [0, 1]. Applied after each layer except last one.
        When set to 0 or None, dropout is disabled.
      dense_norm: whether to use batch normalization after each layer.
      is_training: whether this operation will be used in training or inference.

    Returns:
      `Tensor` with shape `[batch_size, padded_length, d1]`, transformed version of sequence_input.
    """

    input_layers = list(map(lambda u: -u, filter(lambda u: u < 0, dense_layers)))
    if not len(input_layers):
        return sequence_input

    def _input_dnn_producer(flat_input, dense_layers, dense_activation, dense_dropout, dense_norm, is_training):
        with tf.variable_scope('input_dnn', values=(flat_input,)):
            return _build_dnn_layers(
                flat_input=flat_input,
                dense_layers=dense_layers,
                dense_activation=dense_activation,
                dense_dropout=dense_dropout,
                dense_norm=dense_norm,
                is_training=is_training,
            )

    return apply_time_distributed(
        layer_producer=_input_dnn_producer,
        sequence_input=sequence_input,
        dense_layers=input_layers,
        dense_activation=dense_activation,
        dense_dropout=dense_dropout,
        dense_norm=dense_norm,
        is_training=is_training
    )


def build_logits_activations(
        flat_input, logits_size, dense_layers, dense_activation, dense_dropout, dense_norm, is_training):
    """Build a dense network with no activation in last layer.

    Args:
      flat_input: `Tensor` with shape `[batch_size, d0]` representing one of the RNN time steps.
      logits_size: size of output dimension.
      dense_layers: iterable of integer number of hidden units per layer. Only positive values will be used.
      dense_activation: activation function for dense layers. One of `DenseActivation` options or callable.
      dense_dropout: dropout rate, a number between [0, 1]. Applied after each layer except last one.
        When set to 0 or None, dropout is disabled.
      dense_norm: whether to use batch normalization after each layer.
      is_training: whether this operation will be used in training or inference.

    Returns:
      `Tensor` with shape `[batch_size, d1]`, transformed version of flat_input.
    """

    with tf.variable_scope('output_dnn', values=(flat_input,)):
        input_layers = list(filter(lambda u: u > 0, dense_layers))
        flat_input = _build_dnn_layers(
            flat_input=flat_input,
            dense_layers=input_layers,
            dense_activation=dense_activation,
            dense_dropout=dense_dropout,
            dense_norm=dense_norm,
            is_training=is_training,
        )

    with tf.variable_scope('logits_activations', values=(flat_input,)) as logits_scope:
        flat_input = tf.layers.dense(
            flat_input,
            units=logits_size,
            activation=None,
            name=logits_scope
        )

        _add_hidden_layer_summary(flat_input, logits_scope.name)

        return flat_input


def _build_dnn_layers(flat_input, dense_layers, dense_activation, dense_dropout, dense_norm, is_training):
    """Build a dense network layers.

    Args:
      flat_input: `Tensor` with shape `[batch_size, d0]` representing one of the RNN time steps.
      dense_layers: iterable of integer number of hidden units per layer.
      dense_activation: activation function for dense layers. One of `DenseActivation` options or callable.
      dense_dropout: dropout rate, a number between [0, 1]. Applied after each layer except last one.
        When set to 0 or None, dropout is disabled.
      dense_norm: whether to use batch normalization after each layer.
      is_training: whether this operation will be used in training or inference.

    Returns:
      `Tensor` with shape `[batch_size, d1]`, transformed version of flat_input.
    """

    DenseActivation.validate(dense_activation)
    activation_function = DenseActivation.instance(dense_activation)

    for layer_id, num_units in enumerate(dense_layers):
        if num_units <= 0:
            continue

        with tf.variable_scope('layer_%d' % layer_id, values=(flat_input,)) as layer_scope:

            flat_input = tf.layers.dense(
                flat_input,
                units=num_units,
                activation=activation_function,
                name=layer_scope
            )

            if dense_dropout:
                flat_input = tf.layers.dropout(
                    flat_input,
                    rate=dense_dropout,
                    training=is_training,
                    name='dropout_%d' % layer_id,
                )

            if dense_norm:
                flat_input = tf.layers.batch_normalization(
                    flat_input,
                    momentum=0.999,
                    training=is_training,
                    name='batchnorm_%d' % layer_id,
                )

            _add_hidden_layer_summary(flat_input, layer_scope.name)

    return flat_input


def apply_time_distributed(layer_producer, sequence_input, *args, **kwargs):
    """Apply dense layers to sequential input.

    Args:
      layer_producer: Dense neural network producer function.
        Should accept `Tensor` with shape `[batch_size0, d0]` as first argument. An input of dense network.
        Should return `Tensor` with shape `[batch_size0, d1]`. Result activations after applying dense layers to each
            time-step examples.
      sequence_input: `Tensor` with shape `[batch_size, padded_length, d0]`. Each input time step will be represented
        as separate example in layer producer. Dimension `d0` will be constant during all steps.

    Returns:
      `Tensor` with shape `[batch_size, padded_length, d1]` representing the activations of all time steps.
    """

    # with tf.variable_scope('time_distributed', values=(sequence_input,)):

    if not callable(layer_producer):
        raise ValueError('Invalid layer producer type. Expected callable, got {}'.format(type(layer_producer)))

    with tf.variable_scope('time_distributed'):

        # Estimate original dimensions
        sequence_input.get_shape().assert_has_rank(3)

        with tf.name_scope('shape_components'):
            batch_size, padded_length, _ = tf.unstack(tf.shape(sequence_input))
            flat_batch = tf.multiply(batch_size, padded_length)
            _, _, input_units = sequence_input.get_shape().as_list()

        if input_units is None:
            raise ValueError('Last input dimensions should be defined. Got {}'.format(sequence_input.get_shape()))

        # Apply dense layers
        flat_input = tf.reshape(sequence_input, [flat_batch, input_units], name='time_unwrap')

        # Remove time dimension
        flat_output = layer_producer(flat_input, *args, **kwargs)

        # Estimate output shape
        flat_output.get_shape().assert_has_rank(2)
        _, output_units = flat_output.get_shape().as_list()

        if output_units is None:
            raise ValueError('Second output dimension should be defined. Got {}'.format(flat_output.get_shape()))

        # Restore time dimension
        sequence_output = tf.reshape(flat_output, [batch_size, padded_length, output_units], name='time_wrap')

        return sequence_output
