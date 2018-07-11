from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.estimator import binary_classification_head, multi_class_head, regression_head
from tensorflow.contrib.training import HParams
from .rnn import RnnImplementation, RnnDirection, RnnType
from .dense import PredictionType, DenseActivation
from .model import build_model_fn
from .weight import sequence_weights_column


class SequenceEstimator(tf.estimator.Estimator):
    def __init__(self,
                 model_params,
                 sequence_columns,
                 length_column,
                 context_columns=None,
                 weight_column=None,
                 loss_reduction=tf.losses.Reduction.SUM,
                 model_dir=None,
                 warm_start_from=None,
                 config=None):
        """Initializes `SequenceEstimator` instance.

        Args:
          model_params: `dict` with model parameters. Should contain:
              sequence_dropout: Sequence input dropout rate, a number between [0, 1].
                When set to 0 or None, dropout is disabled.
              context_dropout: Context input dropout rate, a number between [0, 1].
                When set to 0 or None, dropout is disabled.
              rnn_implementation: internal RNN implementation. One of `RnnImplementation` option.
              rnn_direction: layers direction. One of `RNNDirection` options.
                Stacked direction available only with regular implementation and more then 1 layers.
              rnn_layers: number of layers.
              rnn_type: type of cell. One of `RNNCell` options.
              rnn_units: number of cells per layer.
              rnn_dropout: dropout rate, a number between [0, 1]. Applied after each RNN layer.
                When set to 0 or None, dropout is disabled.
              dense_layers: iterable of integer number of hidden units per layer.
              dense_activation: name of activation function applied to each dense layer.
                Should be fully defined function path.
              dense_dropout: dropout rate, a number between [0, 1]. Applied after each dense layer.
                When set to 0 or None, dropout is disabled.
              train_optimizer: name of `Optimizer`.
              learning_rate: optimizer learning rate.
          sequence_columns: iterable containing all the feature columns describing sequence features.
            All items in the set should be instances of classes derived from `FeatureColumn`.
          length_column: features key or a `_NumericColumn`. Used as a key to fetch length tensor from features.
          context_columns: iterable containing all feature columns describing context features i.e. features that apply
            across all time steps. All items in the set should be instances of classes derived from `FeatureColumn`.
          weight_column: A string or a `_NumericColumn` created by `tf.feature_column.numeric_column` defining
            feature column representing weights. It is used to down weight or boost examples during training.
            It will be multiplied by the loss of the example. If it is a string, it is used as a key to fetch
            weight tensor from the `features`. If it is a `_NumericColumn`, raw tensor is fetched by key
            `weight_column.key`.
          loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to reduce training loss
            over batch. Defaults to `SUM`.
          model_dir: Directory to save model parameters, graph and etc. This can also be used to load checkpoints
            from the directory into a estimator to continue training a previously saved model.
          warm_start_from: A string filepath to a checkpoint to warm-start from, or a `WarmStartSettings` object
            to fully configure warm-starting.  If the string filepath is provided instead of a `WarmStartSettings`,
            then all weights are warm-started, and it is assumed that vocabularies and Tensor names are unchanged.
          config: `RunConfig` object to configure the runtime settings.
        """

        self.sequence_columns = sequence_columns
        self.length_column = length_column
        self.context_columns = context_columns
        self.weight_column = weight_column
        self.loss_reduction = loss_reduction

        _params = self._model_params(model_params)
        _model_fn = self._model_fn()

        super(SequenceEstimator, self).__init__(_model_fn, model_dir, config, _params, warm_start_from)

    @classmethod
    def _model_params(cls, user_params):
        """Initializes `HParams` instance from default and user-defined model parameters

        Args:
          user_params: `dict` with model parameters. See __init__ for more details.

        Returns:
          `HParams` instance with all required parameters set.
        """
        prediction_type = cls._prediction_type()
        PredictionType.validate(prediction_type)

        params = HParams(
            prediction_type=prediction_type,  # create param and remember type
            sequence_dropout=0.0,
            context_dropout=0.0,
            rnn_implementation=RnnImplementation.REGULAR,
            rnn_direction=RnnDirection.BIDIRECTIONAL,
            rnn_type=RnnType.LSTM,
            rnn_layers=1,
            rnn_units=1,
            rnn_dropout=0.0,
            dense_layers=[0],  # create param and remember type
            dense_activation=DenseActivation.RELU,
            dense_dropout=0.0,
            train_optimizer='Adam',
            learning_rate=0.001,
        )
        params.set_hparam('dense_layers', [])  # set actual default value for param with known type
        params.override_from_dict(user_params)  # update params requested by user
        params.set_hparam('prediction_type', prediction_type)  # disallow override this param

        return params

    @staticmethod
    def _prediction_type():
        """Model prediction type.

        Returns:
          One of PredictionType options.
        """
        raise NotImplementedError('Subclasses must override _prediction_type()')

    def _loss_weight(self):
        """Weight column passed to estimator head

        Returns:
          A None, string or a `_NumericColumn`.
        """
        prediction_type = self._prediction_type()
        PredictionType.validate(prediction_type)

        if PredictionType.SINGLE == prediction_type:
            return self.weight_column  # user weights should be passed directly to loss with single prediction
        else:  # PredictionType.MULTIPLE == prediction_type
            return sequence_weights_column()  # sequence items weights based on actual length and user weights

    def _model_weight(self):
        """Weight column passed to model builder. Used to estimate sequence items weights for per-item predictions.

        Returns:
          A None, string or a `_NumericColumn`.
        """
        prediction_type = self._prediction_type()
        PredictionType.validate(prediction_type)

        if PredictionType.SINGLE == prediction_type:
            return None  # no additional weights estimation required
        else:  # PredictionType.MULTIPLE == prediction_type
            return self.weight_column  # required to estimate sequence items weights

    def _estimator_head(self):
        """Estimator head. Depends on particular task.

        Returns:
          `_Head` instance.
        """
        raise NotImplementedError('Subclasses must override _estimator_head()')

    def _model_fn(self):
        """`Estimator` model function

        Returns:
          Model function.
        """
        def model_fn(features, labels, mode, params, config):
            """Call the shared build_model_fn."""
            return build_model_fn(
                sequence_columns=self.sequence_columns,
                length_column=self.length_column,
                context_columns=self.context_columns,
                weight_column=self._model_weight(),
                head=self._estimator_head(),
                features=features,
                labels=labels,
                mode=mode,
                params=params,
                config=config
            )

        return model_fn


class FullSequenceClassifier(SequenceEstimator):
    """Dynamic-length sequence classifier.
    Estimates one class for a whole sequence.
    """

    def __init__(self, label_vocabulary, *args, **kwargs):
        """Initializes a `FullSequenceClassifier` instance.

        Args:
          label_vocabulary: list of strings represents possible label values.
          *args: positional arguments for SequenceEstimator
          **kwargs: keyword arguments for SequenceEstimator
        """

        self.label_vocabulary = label_vocabulary

        super(FullSequenceClassifier, self).__init__(*args, **kwargs)

    @staticmethod
    def _prediction_type():
        return PredictionType.SINGLE

    def _estimator_head(self):
        if len(self.label_vocabulary) == 2:
            return binary_classification_head(
                weight_column=self._loss_weight(),
                label_vocabulary=self.label_vocabulary,
                loss_reduction=self.loss_reduction
            )
        else:
            return multi_class_head(
                len(self.label_vocabulary),
                weight_column=self._loss_weight(),
                label_vocabulary=self.label_vocabulary,
                loss_reduction=self.loss_reduction
            )


class FullSequenceRegressor(SequenceEstimator):
    """Dynamic-length sequence classifier.
    Estimates one value for a whole sequence.
    """

    def __init__(self, label_dimension, *args, **kwargs):
        """Initializes a `FullSequenceRegressor` instance.

        Args:
          label_dimension: Number of regression targets per example. This is the size of the last labels dimension.
          *args: positional arguments for SequenceEstimator
          **kwargs: keyword arguments for SequenceEstimator
        """

        self.label_dimension = label_dimension

        super(FullSequenceRegressor, self).__init__(*args, **kwargs)

    @staticmethod
    def _prediction_type():
        return PredictionType.SINGLE

    def _estimator_head(self):
        return regression_head(
            weight_column=self._loss_weight(),
            label_dimension=self.label_dimension,
            loss_reduction=self.loss_reduction
        )


class SequenceItemsClassifier(FullSequenceClassifier):
    """Dynamic-length sequence items classifier.
    Estimates one class for each sequence item.
    """

    @staticmethod
    def _prediction_type():
        return PredictionType.MULTIPLE


class SequenceItemsRegressor(FullSequenceRegressor):
    """Dynamic-length sequence items regressor.
    Estimates one value for each sequence item.
    """

    @staticmethod
    def _prediction_type():
        return PredictionType.MULTIPLE

    # TODO: crf