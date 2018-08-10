from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
from tensorflow.python.estimator.canned.optimizers import get_optimizer_instance
from tensorflow.contrib.estimator import clip_gradients_by_norm
from tensorflow.contrib.estimator.python.estimator.rnn import _DEFAULT_CLIP_NORM, _DEFAULT_LEARNING_RATE
from .input import build_sequence_input
from .rnn import build_dynamic_rnn, select_last_activations
from .dense import build_input_dnn, build_logits_activations, apply_time_distributed
from .head import SequenceLengthProvider


class PredictionType(object):
    SINGLE = 'single'
    MULTIPLE = 'multiple'

    @classmethod
    def all(cls):
        return cls.SINGLE, cls.MULTIPLE

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            raise ValueError('Invalid sequence prediction type: {}'.format(key))


def build_model_fn(features, labels, mode, params, config,
                   estimator_head, prediction_type,
                   sequence_columns, context_columns, input_partitioner, sequence_dropout, context_dropout,
                   rnn_type, rnn_layers, rnn_dropout,
                   dense_layers, dense_activation, dense_dropout, dense_norm,
                   train_optimizer='Adam', learning_rate=_DEFAULT_LEARNING_RATE):
    """Combine sequence and context features into sequence input tensor.
    Args:
      features: `dict` containing the input `Tensor`s.
      labels: single `Tensor` or `dict` of same (for multi-head models).
      mode: specifies if this training, evaluation or prediction.
      config: optional configuration object.
      estimator_head: `_Head` estimator instance.
      prediction_type: whether all time steps should be used or just last one. One of `PredictionType` options.
      sequence_columns: iterable containing `FeatureColumn`s that represent sequential model inputs.
      context_columns: iterable containing `FeatureColumn`s that represent model inputs not associated with a
        specific timestep.
      input_partitioner: partitioner for input layer variables.
      sequence_dropout: sequence input dropout rate, a number between [0, 1].
        When set to 0 or None, dropout is disabled.
      context_dropout: context input dropout rate, a number between [0, 1].
        When set to 0 or None, dropout is disabled.
      rnn_type: type, direction and implementations of RNN. One of `RnnType` options.
      rnn_layers: iterable of integer number of hidden units per layer.
      rnn_dropout: recurrent layers dropout rate, a number between [0, 1]. Applied after each layer.
        When set to 0 or None, dropout is disabled.
      dense_layers: iterable of integer number of hidden units per layer. Negative values corresponding to layers
        before RNN, postivie right after.
      dense_activation: activation function for dense layers. One of `DenseActivation` options or callable.
        Should be fully defined function path.
      dense_dropout: dropout rate, a number between [0, 1]. Applied after each layer except last one.
        When set to 0 or None, dropout is disabled.
      dense_norm: whether to use batch normalization after each layer.
      train_optimizer: string or `Optimizer` object, or callable that creates the optimizer to use for training.
        If not specified, will use the Adam optimizer with a default learning rate of 0.05.
      learning_rate: floating point value. The learning rate.
    Returns:
      `EstimatorSpec`
    """
    if not isinstance(features, dict):
        raise ValueError('Features should be a dictionary of `Tensor`s. Given type: {}'.format(type(features)))

    # Create input features partitioner
    num_ps_replicas = config.num_ps_replicas if config else 0
    partitioner = tf.min_max_variable_partitioner(max_partitions=num_ps_replicas)

    with tf.variable_scope('model', values=tuple(six.itervalues(features)), partitioner=partitioner):
        is_training = tf.estimator.ModeKeys.TRAIN == mode

        input_partitioner = input_partitioner or tf.min_max_variable_partitioner(
            max_partitions=num_ps_replicas,
            min_slice_size=64 << 20
        )

        # Transform input features
        sequence_input, sequence_length = build_sequence_input(
            features=features,
            sequence_columns=sequence_columns,
            context_columns=context_columns,
            input_partitioner=input_partitioner,
            sequence_dropout=sequence_dropout,
            context_dropout=context_dropout,
            is_training=is_training,
        )

        # Add input dense layers
        sequence_input = build_input_dnn(
            sequence_input=sequence_input,
            dense_layers=dense_layers,
            dense_activation=dense_activation,
            dense_dropout=dense_dropout,
            dense_norm=dense_norm,
            is_training=is_training
        )

        # Add recurrent layers
        rnn_outputs = build_dynamic_rnn(
            sequence_input=sequence_input,
            sequence_length=sequence_length,
            rnn_type=rnn_type,
            rnn_layers=rnn_layers,
            rnn_dropout=rnn_dropout,
            is_training=is_training,
        )

        PredictionType.validate(prediction_type)
        if PredictionType.SINGLE == prediction_type:
            # Extract last non-padded output
            last_output = select_last_activations(
                rnn_outputs=rnn_outputs,
                sequence_length=sequence_length,
            )

            # Add dense layers
            logits = build_logits_activations(
                flat_input=last_output,
                logits_size=estimator_head.logits_dimension,
                dense_layers=dense_layers,
                dense_activation=dense_activation,
                dense_dropout=dense_dropout,
                dense_norm=dense_norm,
                is_training=is_training
            )

        else:
            assert PredictionType.MULTIPLE == prediction_type

            # Add time-distributed dense layers
            logits = apply_time_distributed(
                layer_producer=build_logits_activations,
                sequence_input=rnn_outputs,
                logits_size=estimator_head.logits_dimension,
                dense_layers=dense_layers,
                dense_activation=dense_activation,
                dense_dropout=dense_dropout,
                dense_norm=dense_norm,
                is_training=is_training
            )

            if not isinstance(estimator_head, SequenceLengthProvider):
                raise ValueError('Estimator `_Head` for multiple predictions '
                                 'should be successor of SequenceLengthContainer')
            estimator_head.set_sequence_length(sequence_length)

    # Create optimizer instance
    optimizer = get_optimizer_instance(train_optimizer, learning_rate=learning_rate)
    optimizer = clip_gradients_by_norm(optimizer, _DEFAULT_CLIP_NORM)

    return estimator_head.create_estimator_spec(
        features=features,
        mode=mode,
        logits=logits,
        labels=labels,
        optimizer=optimizer,
    )
