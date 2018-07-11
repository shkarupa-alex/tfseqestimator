from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.estimator.canned.optimizers import get_optimizer_instance
from .input import build_sequence_input
from .rnn import build_dynamic_rnn
from .dense import build_logits_activations, PredictionType
from .weight import make_sequence_weights


def build_model_fn(sequence_columns, length_column, context_columns, weight_column, head, features, labels, mode,
                   params, config):
    """Combine sequence and context features into sequence input tensor.
    Args:
      sequence_columns: An iterable containing all the feature columns describing sequence features.
        All items in the set should be instances of classes derived from `FeatureColumn`.
      length_column: string or a `_NumericColumn` created by `tf.feature_column.numeric_column` defining
        feature column representing real sequence length.
        If it is a string, it is used as a key to fetch length tensor from the `features`.
        If it is a `_NumericColumn`, raw tensor is fetched by key `length_column.key`.
      context_columns: An iterable containing all the feature columns describing context features i.e. features that
        apply across all time steps. All items in the set should be instances of classes derived from `FeatureColumn`.
      head: `_Head` estimator instance.
      features: `dict` containing input and sequence length information.
      labels: single `Tensor` or `dict` of same (for multi-head models).
      mode: Specifies if this training, evaluation or prediction.
      params: `HParams` instance with model parameters. Should contain:
          prediction_type: whether all time steps should be used or just last one. One of `PredictionType` options.
          sequence_dropout: Sequence input dropout rate, a number between [0, 1].
            When set to 0 or None, dropout is disabled.
          context_dropout: Context input dropout rate, a number between [0, 1].
            When set to 0 or None, dropout is disabled.
          rnn_implementation: internal implementation. One of `RNNArchitecture` options.
          rnn_direction: layers direction. One of `RNNDirection` options.
            Stacked direction available only with regular implementation and 2+ layers.
          rnn_layers: number of layers.
          rnn_type: type of cell. One of `RNNCell` options.
          rnn_units: number of cells per layers.
          rnn_dropout: dropout rate, a number between [0, 1]. Applied after each layer.
            When set to 0 or None, dropout is disabled.
          dense_layers: iterable of integer number of hidden units per layer.
          dense_activation: activation function applied to each layer except last one.
          dense_dropout: dropout rate, a number between [0, 1]. Applied after each layer except last one.
            When set to 0 or None, dropout is disabled.
      config: Optional configuration object.
    Returns:
      `EstimatorSpec`
    """

    is_training = tf.estimator.ModeKeys.TRAIN == mode

    # Transform input features
    sequence_input, sequence_length = build_sequence_input(
        sequence_columns, length_column, context_columns, features, params, is_training)

    # Add recurrent layers
    rnn_outputs, last_output = build_dynamic_rnn(sequence_input, sequence_length, params, is_training)
    # _labels = tf.Print(labels, [rnn_outputs], summarize=100)

    # Add dense layers
    logits = build_logits_activations(rnn_outputs, last_output, params, head.logits_dimension, is_training)

    # Create optimizer instance
    optimizer = get_optimizer_instance(params.train_optimizer, learning_rate=params.learning_rate)

    PredictionType.validate(params.prediction_type)
    if PredictionType.SINGLE == params.prediction_type:
        _features = features
    else:  # PredictionType.MULTIPLE == params.prediction_type
        _features = make_sequence_weights(features, weight_column, logits, sequence_length)

    return head.create_estimator_spec(
        features=_features,
        mode=mode,
        logits=logits,
        labels=labels,
        optimizer=optimizer,
    )
