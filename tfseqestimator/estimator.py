from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.estimator.canned.head import _Head
from tensorflow.contrib.estimator import binary_classification_head, multi_class_head, regression_head
from tensorflow.contrib.estimator.python.estimator.rnn import _DEFAULT_LEARNING_RATE
from .rnn import RnnType
from .dense import DenseActivation
from .model import build_model_fn, PredictionType
from .head import sequence_binary_classification_head_with_sigmoid, sequence_multi_class_head_with_softmax
from .head import sequence_regression_head_with_mse_loss


class SequenceEstimator(tf.estimator.Estimator):
    """Dynamic-length sequence estimator with user-specified head and prediction type."""

    def __init__(
            self, estimator_head, prediction_type,
            sequence_columns, context_columns=None, input_partitioner=None, sequence_dropout=0., context_dropout=0.,
            rnn_type=RnnType.REGULAR_FORWARD_GRU, rnn_layers=None, rnn_dropout=0.,
            dense_layers=None, dense_activation=DenseActivation.RELU, dense_dropout=0., dense_norm=False,
            train_optimizer='Adam', learning_rate=_DEFAULT_LEARNING_RATE,
            model_dir=None, warm_start_from=None, config=None):
        """Initializes `SequenceEstimator` instance.

        Args:
          estimator_head: `_Head` instance constructed with a method such as `tf.contrib.estimator.multi_label_head`.
            Specifies the model's output and loss function to be optimized.
          prediction_type: one of PredictionType options.
            Specifies if full sequence should be predicted or each item separately.
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
          dense_dropout: dropout rate, a number between [0, 1]. Applied after each layer except last one.
            When set to 0 or None, dropout is disabled.
          dense_norm: whether to use batch normalization after each dense layer.
          train_optimizer: string or `Optimizer` object, or callable that creates the optimizer to use for training.
            If not specified, will use the Adam optimizer with a default learning rate of 0.05.
          learning_rate: floating point value. The learning rate.
          model_dir: directory to save model parameters, graph and etc. This can also be used to load
            checkpoints from the directory into a estimator to continue training a previously saved model.
          warm_start_from: string filepath to a checkpoint to warm-start from, or a `WarmStartSettings`
            object to fully configure warm-starting.  If the string filepath is provided instead of a
            `WarmStartSettings`, then all weights are warm-started, and it is assumed that vocabularies and Tensor
            names are unchanged.
          config: optional. `RunConfig` object to configure the runtime settings.
        """

        if not isinstance(estimator_head, _Head):
            raise ValueError('Invalid estimator head type: {}'.format(type(estimator_head)))
        self.estimator_head = estimator_head

        PredictionType.validate(prediction_type)
        self.prediction_type = prediction_type

        self.sequence_columns = sequence_columns
        self.context_columns = context_columns
        self.input_partitioner = input_partitioner
        self.sequence_dropout = sequence_dropout
        self.context_dropout = context_dropout
        self.rnn_type = rnn_type
        self.rnn_layers = [1] if rnn_layers is None else list(rnn_layers)
        self.rnn_dropout = rnn_dropout
        self.dense_layers = [] if dense_layers is None else list(dense_layers)
        self.dense_activation = dense_activation
        self.dense_dropout = dense_dropout
        self.dense_norm=dense_norm
        self.train_optimizer = train_optimizer
        self.learning_rate = learning_rate

        super(SequenceEstimator, self).__init__(self._model_fn, model_dir, config, None, warm_start_from)

    def _model_fn(self, features, labels, mode, params, config):
        """`Estimator` model function.

        Returns:
          Model function.
        """
        return build_model_fn(
            features=features,
            labels=labels,
            mode=mode,
            params=params,
            config=config,
            estimator_head=self.estimator_head,
            prediction_type=self.prediction_type,
            sequence_columns=self.sequence_columns,
            context_columns=self.context_columns,
            input_partitioner=self.input_partitioner,
            sequence_dropout=self.sequence_dropout,
            context_dropout=self.context_dropout,
            rnn_type=self.rnn_type,
            rnn_layers=self.rnn_layers,
            rnn_dropout=self.rnn_dropout,
            dense_norm=self.dense_norm,
            dense_layers=self.dense_layers,
            dense_activation=self.dense_activation,
            dense_dropout=self.dense_dropout,
            train_optimizer=self.train_optimizer,
            learning_rate=self.learning_rate,
        )


class FullSequenceClassifier(SequenceEstimator):
    """Dynamic-length sequence classifier. Estimates one class for a whole sequence."""

    def __init__(self, label_vocabulary, loss_reduction=tf.losses.Reduction.SUM, weight_column=None, *args, **kwargs):
        """Initializes a `FullSequenceClassifier` instance.

        Args:
          label_vocabulary: list of strings represents possible label values.
          loss_reduction: Optional. One of `tf.losses.Reduction` except `NONE`. Describes how to reduce training loss
            over batch. Defaults to `SUM`.
          weight_column: Optional. String key used to fetch user-defined weights `Tensor` from the `features`.
          *args: positional arguments for SequenceEstimator
          **kwargs: keyword arguments for SequenceEstimator
        """

        self.label_vocabulary = label_vocabulary
        self.loss_reduction = loss_reduction

        estimator_head = self._estimator_head(weight_column)
        prediction_type = self._prediction_type()

        super(FullSequenceClassifier, self).__init__(
            estimator_head=estimator_head,
            prediction_type=prediction_type,
            *args,
            **kwargs
        )

    def _estimator_head(self, weight_column):
        if len(self.label_vocabulary) == 2:
            return binary_classification_head(
                weight_column=weight_column,
                label_vocabulary=self.label_vocabulary,
                loss_reduction=self.loss_reduction,
            )

        return multi_class_head(
            len(self.label_vocabulary),
            weight_column=weight_column,
            label_vocabulary=self.label_vocabulary,
            loss_reduction=self.loss_reduction,
        )

    @staticmethod
    def _prediction_type():
        return PredictionType.SINGLE


class FullSequenceRegressor(SequenceEstimator):
    """Dynamic-length sequence regressor. Estimates one value for a whole sequence."""

    def __init__(self, label_dimension, loss_reduction=tf.losses.Reduction.SUM, weight_column=None, *args, **kwargs):
        """Initializes a `FullSequenceRegressor` instance.

        Args:
          label_dimension: Number of regression targets per example. This is the size of the last labels dimension.
          loss_reduction: Optional. One of `tf.losses.Reduction` except `NONE`. Describes how to reduce training loss
            over batch. Defaults to `SUM`.
          weight_column: Optional. String key used to fetch user-defined weights `Tensor` from the `features`.
          *args: positional arguments for SequenceEstimator
          **kwargs: keyword arguments for SequenceEstimator
        """

        self.label_dimension = label_dimension
        self.loss_reduction = loss_reduction

        estimator_head = self._estimator_head(weight_column)
        prediction_type = self._prediction_type()

        super(FullSequenceRegressor, self).__init__(
            estimator_head=estimator_head,
            prediction_type=prediction_type,
            *args,
            **kwargs
        )

    def _estimator_head(self, weight_column):
        return regression_head(
            weight_column=weight_column,
            label_dimension=self.label_dimension,
            loss_reduction=self.loss_reduction,
        )

    @staticmethod
    def _prediction_type():
        return PredictionType.SINGLE


class SequenceItemsClassifier(FullSequenceClassifier):
    """Dynamic-length sequence items classifier. Estimates one class for each sequence item."""

    def _estimator_head(self, weight_column):
        if len(self.label_vocabulary) == 2:
            return sequence_binary_classification_head_with_sigmoid(
                weight_column=weight_column,
                label_vocabulary=self.label_vocabulary,
                loss_reduction=self.loss_reduction,
            )

        return sequence_multi_class_head_with_softmax(
            len(self.label_vocabulary),
            weight_column=weight_column,
            label_vocabulary=self.label_vocabulary,
            loss_reduction=self.loss_reduction,
        )

    @staticmethod
    def _prediction_type():
        return PredictionType.MULTIPLE


class SequenceItemsRegressor(FullSequenceRegressor):
    """Dynamic-length sequence items regressor. Estimates one value for each sequence item."""

    def _estimator_head(self, weight_column):
        return sequence_regression_head_with_mse_loss(
            weight_column=weight_column,
            label_dimension=self.label_dimension,
            loss_reduction=self.loss_reduction,
        )

    @staticmethod
    def _prediction_type():
        return PredictionType.MULTIPLE
