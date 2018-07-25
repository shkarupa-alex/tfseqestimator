from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
from tensorflow.python.estimator.canned.optimizers import get_optimizer_instance
from tensorflow.contrib.estimator import clip_gradients_by_norm
from tensorflow.contrib.estimator.python.estimator.rnn import _DEFAULT_CLIP_NORM
from .input import build_sequence_input
from .rnn import build_dynamic_rnn, select_last_activations
from .dense import build_logits_activations, apply_time_distributed
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


def build_model_fn(
        estimator_head, prediction_type, sequence_columns, context_columns, input_partitioner,
        features, labels, mode, params, config):
    """Combine sequence and context features into sequence input tensor.
    Args:
      estimator_head: `_Head` estimator instance.
      prediction_type: one of PredictionType options.
      sequence_columns: iterable containing `FeatureColumn`s that represent sequential model inputs.
      context_columns: iterable containing `FeatureColumn`s that represent model inputs not associated with a
        specific timestep.
      input_partitioner: Partitioner for input layer variables.
      features: `dict` containing the input `Tensor`s.
      labels: single `Tensor` or `dict` of same (for multi-head models).
      mode: Specifies if this training, evaluation or prediction.
      params: `HParams` instance with model parameters. Should contain:
        prediction_type: whether all time steps should be used or just last one. One of `PredictionType` options.
        sequence_dropout: sequence input dropout rate, a number between [0, 1].
          When set to 0 or None, dropout is disabled.
        context_dropout: context input dropout rate, a number between [0, 1].
          When set to 0 or None, dropout is disabled.
        rnn_type: type, direction and implementations of RNN. One of `RnnType` options.
        rnn_layers: iterable of integer number of hidden units per layer.
        rnn_dropout: recurrent layers dropout rate, a number between [0, 1]. Applied after each layer.
          When set to 0 or None, dropout is disabled.
        dense_layers: iterable of integer number of hidden units per layer.
        dense_activation: name of activation function applied to each dense layer.
          Should be fully defined function path.
        dense_dropout: dropout rate, a number between [0, 1]. Applied after each layer except last one.
          When set to 0 or None, dropout is disabled.
      config: Optional configuration object.
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
            sequence_columns=sequence_columns,
            context_columns=context_columns,
            input_partitioner=input_partitioner,
            features=features,
            params=params,
            is_training=is_training
        )

        # Add recurrent layers
        rnn_outputs = build_dynamic_rnn(
            sequence_input=sequence_input,
            sequence_length=sequence_length,
            params=params,
            is_training=is_training
        )

        PredictionType.validate(prediction_type)
        if PredictionType.SINGLE == prediction_type:
            # Extract last non-padded output
            last_output = select_last_activations(rnn_outputs, sequence_length)

            # Add dense layers
            logits = build_logits_activations(last_output, params, estimator_head.logits_dimension, is_training)

        else:  # PredictionType.MULTIPLE == prediction_type
            # Add time-distributed dense layers
            logits = apply_time_distributed(
                build_logits_activations, rnn_outputs, params, estimator_head.logits_dimension, is_training)

            if not isinstance(estimator_head, SequenceLengthProvider):
                err_msg = 'Estimator `_Head` for multiple predictions should be successor of SequenceLengthContainer'
                raise ValueError(err_msg)
            estimator_head.set_sequence_length(sequence_length)

    # Create optimizer instance
    optimizer = get_optimizer_instance(params.train_optimizer, learning_rate=params.learning_rate)
    optimizer = clip_gradients_by_norm(optimizer, _DEFAULT_CLIP_NORM)

    return estimator_head.create_estimator_spec(
        features=features,
        mode=mode,
        logits=logits,
        labels=labels,
        optimizer=optimizer,
    )
