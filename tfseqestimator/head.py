from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.estimator.python.estimator import multi_head
from tensorflow.python.estimator.canned import head
from tensorflow.python.ops import lookup_ops
from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.estimator.export import export_output
from tensorflow.contrib import crf
from .weight import make_sequence_weights, mask_real_sequence


def sequence_regression_head_with_mse_loss(
        weight_column=None,
        label_dimension=1,
        loss_reduction=tf.losses.Reduction.SUM,
        loss_fn=None,
        inverse_link_fn=None,
        name=None):
    """Creates a `_Head` for sequence regression using the `mean_squared_error` loss.

    The loss is the weighted sum over all input dimensions. Namely, if the input labels have shape
    `[batch_size, label_dimension]`, the loss is the weighted sum over both `batch_size` and `label_dimension`.

    The head expects `logits` with shape `[D0, D1, ... DN, label_dimension]`. In many applications, the shape is
    `[batch_size, label_dimension]`.

    The `labels` shape must match `logits`, namely `[D0, D1, ... DN, label_dimension]`. If `label_dimension=1`, shape
    `[D0, D1, ... DN]` is also supported.

    If `weight_column` is specified, weights must be of shape `[D0, D1, ... DN]`, `[D0, D1, ... DN, 1]` or
    `[D0, D1, ... DN, label_dimension]`.

    Supports custom `loss_fn`. `loss_fn` takes `(labels, logits)` or `(labels, logits, features)` as arguments and
    returns unreduced loss with shape `[D0, D1, ... DN, label_dimension]`.

    Also supports custom `inverse_link_fn`, also known as 'mean function'. `inverse_link_fn` takes `logits` as
    argument and returns predicted values. This function is the inverse of the link function defined in
    https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function Namely, for poisson regression, set
    `inverse_link_fn=tf.exp`.

    Args:
      weight_column: A string or a `_NumericColumn` created by `tf.feature_column.numeric_column` defining feature
        column representing weights. It is used to down weight or boost examples during training. It will be multiplied
        by the loss of the example.
      label_dimension: Number of regression labels per example. This is the size of the last dimension of the labels
        `Tensor` (typically, this has shape `[batch_size, label_dimension]`).
      loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to reduce training loss over batch.
        Defaults to `SUM`.
      loss_fn: Optional loss function. Defaults to `mean_squared_error`.
      inverse_link_fn: Optional inverse link function, also known as 'mean function'. Defaults to identity.
      name: name of the head. If provided, summary and metrics keys will be suffixed by `"/" + name`. Also used as
        `name_scope` when creating ops.

    Returns:
      An instance of `_Head` for linear regression.
    """

    if loss_reduction not in tf.losses.Reduction.all() or loss_reduction == tf.losses.Reduction.NONE:
        raise ValueError('Invalid loss_reduction: {}'.format(loss_reduction))
    if loss_fn:
        head._validate_loss_fn_args(loss_fn)

    return _RegressionHeadWithMeanSquaredErrorLoss(
        weight_column=weight_column,
        label_dimension=label_dimension,
        loss_reduction=loss_reduction,
        loss_fn=loss_fn,
        inverse_link_fn=inverse_link_fn,
        name=name)


def sequence_binary_classification_head_with_sigmoid(
        weight_column=None,
        thresholds=None,
        label_vocabulary=None,
        loss_reduction=tf.losses.Reduction.SUM,
        loss_fn=None,
        name=None):
    """Creates a `_Head` for sequence single label binary classification.

    This head uses `sigmoid_cross_entropy_with_logits` loss.

    The head expects `logits` with shape `[D0, D1, ... DN, 1]`. In many applications, the shape is `[batch_size, 1]`.

    `labels` must be a dense `Tensor` with shape matching `logits`, namely `[D0, D1, ... DN, 1]`. If `label_vocabulary`
    given, `labels` must be a string `Tensor` with values from the vocabulary. If `label_vocabulary` is not given,
    `labels` must be float `Tensor` with values in the interval `[0, 1]`.

    If `weight_column` is specified, weights must be of shape `[D0, D1, ... DN]`, or `[D0, D1, ... DN, 1]`.

    The loss is the weighted sum over the input dimensions. Namely, if the input labels have shape `[batch_size, 1]`,
    the loss is the weighted sum over `batch_size`.

    Also supports custom `loss_fn`. `loss_fn` takes `(labels, logits)` or `(labels, logits, features)` as arguments and
    returns unreduced loss with shape `[D0, D1, ... DN, 1]`. `loss_fn` must support float `labels` with shape
    `[D0, D1, ... DN, 1]`. Namely, the head applies `label_vocabulary` to the input labels before passing them to
    `loss_fn`.

    Args:
      weight_column: A string or a `_NumericColumn` created by `tf.feature_column.numeric_column` defining feature
        column representing weights. It is used to down weight or boost examples during training. It will be multiplied
        by the loss of the example.
      thresholds: Iterable of floats in the range `(0, 1)`. For binary classification metrics such as precision and
        recall, an eval metric is generated for each threshold value. This threshold is applied to the logistic values
        to determine the binary classification (i.e., above the threshold is `true`, below is `false`.
      label_vocabulary: A list or tuple of strings representing possible label values. If it is not given, that means
        labels are already encoded within [0, 1]. If given, labels must be string type and have any value in
        `label_vocabulary`. Note that errors will be raised if `label_vocabulary` is not provided but labels are
        strings.
      loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to reduce training loss over batch.
        Defaults to `SUM`.
      loss_fn: Optional loss function.
      name: name of the head. If provided, summary and metrics keys will be suffixed by `"/" + name`. Also used as
        `name_scope` when creating ops.

    Returns:
      An instance of `_Head` for binary classification.
    """

    thresholds = tuple(thresholds) if thresholds else tuple()
    if label_vocabulary is not None and not isinstance(label_vocabulary, (list, tuple)):
        raise TypeError('label_vocabulary should be a list or tuple. Given type: {}'.format(type(label_vocabulary)))

    for threshold in thresholds:
        if (threshold <= 0.0) or (threshold >= 1.0):
            raise ValueError('thresholds not in (0, 1): {}.'.format((thresholds,)))
    if loss_reduction not in tf.losses.Reduction.all() or loss_reduction == tf.losses.Reduction.NONE:
        raise ValueError('Invalid loss_reduction: {}'.format(loss_reduction))
    if loss_fn:
        head._validate_loss_fn_args(loss_fn)

    return _BinaryLogisticHeadWithSigmoidCrossEntropyLoss(
        weight_column=weight_column,
        thresholds=thresholds,
        label_vocabulary=label_vocabulary,
        loss_reduction=loss_reduction,
        loss_fn=loss_fn,
        name=name)


def sequence_multi_class_head_with_softmax(
        n_classes,
        weight_column=None,
        label_vocabulary=None,
        loss_reduction=tf.losses.Reduction.SUM,
        loss_fn=None,
        name=None):
    """Creates a '_Head' for sequence multi class classification.

    The head expects `logits` with shape `[D0, D1, ... DN, n_classes]`. In many applications, the shape is
    `[batch_size, n_classes]`.

    `labels` must be a dense `Tensor` with shape matching `logits`, namely `[D0, D1, ... DN, 1]`. If `label_vocabulary`
    given, `labels` must be a string `Tensor` with values from the vocabulary. If `label_vocabulary` is not given,
    `labels` must be an integer `Tensor` with values specifying the class index.

    If `weight_column` is specified, weights must be of shape `[D0, D1, ... DN]`, or `[D0, D1, ... DN, 1]`.

    The loss is the weighted sum over the input dimensions. Namely, if the input labels have shape `[batch_size, 1]`,
    the loss is the weighted sum over `batch_size`.

    Also supports custom `loss_fn`. `loss_fn` takes `(labels, logits)` or `(labels, logits, features)` as arguments
    and returns unreduced loss with shape `[D0, D1, ... DN, 1]`. `loss_fn` must support integer `labels` with shape
    `[D0, D1, ... DN, 1]`. Namely, the head applies `label_vocabulary` to the input labels before passing them to
    `loss_fn`.

    Args:
      n_classes: Number of classes, must be greater than 2 (for 2 classes, use
        `_BinaryLogisticHeadWithSigmoidCrossEntropyLoss`).
      weight_column: A string or a `_NumericColumn` created by `tf.feature_column.numeric_column` defining feature
        column representing weights. It is used to down weight or boost examples during training. It will be multiplied
        by the loss of the example.
      label_vocabulary: A list or tuple of strings representing possible label values. If it is not given, that means
        labels are already encoded as an integer within [0, n_classes). If given, labels must be of string type and
        have any value in `label_vocabulary`. Note that errors will be raised if `label_vocabulary` is not provided but
        labels are strings.
      loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to reduce training loss over batch.
        Defaults to `SUM`.
      loss_fn: Optional loss function.
      name: name of the head. If provided, summary and metrics keys will be suffixed by `"/" + name`. Also used as
        `name_scope` when creating ops.

    Returns:
      An instance of `_Head` for multi class classification.
    """

    if label_vocabulary is not None and not isinstance(label_vocabulary, (list, tuple)):
        raise ValueError('label_vocabulary should be a list or a tuple. Given type: {}'.format(type(label_vocabulary)))
    if loss_reduction not in tf.losses.Reduction.all() or loss_reduction == tf.losses.Reduction.NONE:
        raise ValueError('Invalid loss_reduction: {}'.format(loss_reduction))
    if loss_fn:
        head._validate_loss_fn_args(loss_fn)

    return _MultiClassHeadWithSoftmaxCrossEntropyLoss(
        n_classes=n_classes,
        weight_column=weight_column,
        label_vocabulary=label_vocabulary,
        loss_reduction=loss_reduction,
        loss_fn=loss_fn,
        name=name)


def sequence_multi_class_head_with_crf_loss(
        n_classes,
        weight_column=None,
        label_vocabulary=None,
        loss_reduction=tf.losses.Reduction.SUM,
        name=None):
    """Creates a '_Head' for sequence single label classification.

    The head expects `logits` with shape `[D0, D1, ... DN, n_classes]`. In many applications, the shape is
    `[batch_size, n_classes]`.

    `labels` must be a dense `Tensor` with shape matching `logits`, namely `[D0, D1, ... DN, 1]`. If `label_vocabulary`
    given, `labels` must be a string `Tensor` with values from the vocabulary. If `label_vocabulary` is not given,
    `labels` must be an integer `Tensor` with values specifying the class index.

    If `weight_column` is specified, weights must be of shape `[D0, D1, ... DN]`, or `[D0, D1, ... DN, 1]`.

    The loss is the weighted sum over the input dimensions. Namely, if the input labels have shape `[batch_size, 1]`,
    the loss is the weighted sum over `batch_size`.

    Args:
      n_classes: Number of classes.
      weight_column: A string or a `_NumericColumn` created by `tf.feature_column.numeric_column` defining feature
        column representing weights. It is used to down weight or boost examples during training. It will be multiplied
        by the loss of the example.
      label_vocabulary: A list or tuple of strings representing possible label values. If it is not given, that means
        labels are already encoded as an integer within [0, n_classes). If given, labels must be of string type and
        have any value in `label_vocabulary`. Note that errors will be raised if `label_vocabulary` is not provided but
        labels are strings.
      loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to reduce training loss over batch.
        Defaults to `SUM`.
      name: name of the head. If provided, summary and metrics keys will be suffixed by `"/" + name`. Also used as
        `name_scope` when creating ops.

    Returns:
      An instance of `_Head` for multi class classification.
    """

    if label_vocabulary is not None and not isinstance(label_vocabulary, (list, tuple)):
        raise ValueError('label_vocabulary should be a list or a tuple. Given type: {}'.format(type(label_vocabulary)))
    if loss_reduction not in tf.losses.Reduction.all() or loss_reduction == tf.losses.Reduction.NONE:
        raise ValueError('Invalid loss_reduction: {}'.format(loss_reduction))

    return _SingleClassHeadWithCrfLogLikelihoodLoss(
        n_classes=n_classes,
        weight_column=weight_column,
        label_vocabulary=label_vocabulary,
        loss_reduction=loss_reduction,
        name=name)


def sequence_multi_head(heads, head_weights=None):
    """Creates a `_Head` for multi-objective sequence learning.

    This class merges the output of multiple `_Head` objects.
    Specifically:
    * For training, sums losses of each head, calls `train_op_fn` with this final loss.
    * For eval, merges metrics by adding `head.name` suffix to the keys in eval metrics, such as `precision/head1`,
        `precision/head2`.
    * For prediction, merges predictions and updates keys in prediction dict to a 2-tuple,
        `(head.name, prediction_key)`. Merges `export_outputs` such that by default the first head is served.

    Args:
      heads: List or tuple of `_Head` instances. All heads must have `name` specified. The first head in the list is
        the default used at serving time.
      head_weights: Optional list of weights, same length as `heads`. Used when merging losses to calculate the
        weighted sum of losses from each head. If `None`, all losses are weighted equally.

    Returns:
      A instance of `_Head` that merges multiple heads.
    """
    if head_weights and len(head_weights) != len(heads):
        raise ValueError(
            'heads and head_weights must have the same size. '
            'Given len(heads): {}. Given len(head_weights): {}.'.format(len(heads), len(head_weights)))
    if not heads:
        raise ValueError('Must specify heads. Given: {}'.format(heads))
    for head in heads:
        if not head.name:
            raise ValueError('All given heads must have name specified. ' 'Given: {}'.format(head))
        if not isinstance(head, SequenceLengthProvider):
            raise ValueError('All given heads must be successor of SequenceLengthContainer')

    return _MultiHead(
        heads=tuple(heads),
        head_weights=tuple(head_weights) if head_weights else tuple())


class SequenceLengthProvider:
    """Stores and provides sequence length `Tensor`."""

    _sequence_length = None

    def set_sequence_length(self, actual_length):
        """Remembers actual sequences length.

        Args:
          actual_length: integer `Tensor` with shape `[batch_size]`.
        """

        self._sequence_length = tf.convert_to_tensor(actual_length)
        self._sequence_length.get_shape().assert_has_rank(1)

    def _get_sequence_length(self):
        """Actual sequences length.

        Returns:
          integer `Tensor` with shape `[batch_size]`.
        """

        if self._sequence_length is None:
            raise ValueError('Sequence length should be provided with "set_sequence_length" call')

        return self._sequence_length


class _RegressionHeadWithMeanSquaredErrorLoss(
    head._RegressionHeadWithMeanSquaredErrorLoss, SequenceLengthProvider):
    """See `sequence_regression_head_with_mse_loss`."""

    def create_loss(self, features, mode, logits, labels):
        """See `Head`."""
        del mode  # Unused for this head.

        logits = tf.convert_to_tensor(logits)
        logits.get_shape().assert_has_rank(3)

        labels = head._check_dense_labels_match_logits_and_reshape(
            labels=labels, logits=logits, expected_labels_dimension=self._logits_dimension)
        labels.get_shape().assert_has_rank(3)
        labels = tf.to_float(labels)

        if self._loss_fn:
            unweighted_loss = head._call_loss_fn(
                loss_fn=self._loss_fn, labels=labels, logits=logits,
                features=features, expected_loss_dim=self._logits_dimension)
        else:
            unweighted_loss = tf.losses.mean_squared_error(
                labels=labels, predictions=logits, reduction=tf.losses.Reduction.NONE)

        weights = head._get_weights_and_check_match_logits(
            features=features, weight_column=self._weight_column, logits=logits, allow_per_logit_weights=True)

        sequence_length = self._get_sequence_length()
        sequence_weights = tf.convert_to_tensor(weights)  # _get_weights_and_check_match_logits may return scalar
        sequence_weights = make_sequence_weights(sequence_weights, logits, sequence_length)

        masked_loss = mask_real_sequence(unweighted_loss, sequence_length)
        training_loss = tf.losses.compute_weighted_loss(
            masked_loss, weights=sequence_weights, reduction=self._loss_reduction)

        return head.LossSpec(
            training_loss=training_loss,
            unreduced_loss=masked_loss,
            weights=sequence_weights,
            processed_labels=labels)


class _BinaryLogisticHeadWithSigmoidCrossEntropyLoss(
    head._BinaryLogisticHeadWithSigmoidCrossEntropyLoss, SequenceLengthProvider):
    def create_loss(self, features, mode, logits, labels):
        """See `Head`."""
        del mode  # Unused for this head.

        logits = tf.convert_to_tensor(logits)
        logits.get_shape().assert_has_rank(3)

        labels = head._check_dense_labels_match_logits_and_reshape(
            labels=labels, logits=logits, expected_labels_dimension=1)
        if self._label_vocabulary is not None:
            labels = lookup_ops.index_table_from_tensor(
                vocabulary_list=tuple(self._label_vocabulary), name='class_id_lookup').lookup(labels)
        labels.get_shape().assert_has_rank(3)
        labels = tf.to_float(labels)
        labels = head._assert_range(labels, n_classes=2)

        if self._loss_fn:
            unweighted_loss = head._call_loss_fn(
                loss_fn=self._loss_fn, labels=labels, logits=logits,
                features=features, expected_loss_dim=1)
        else:
            unweighted_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=logits)

        weights = head._get_weights_and_check_match_logits(
            features=features, weight_column=self._weight_column, logits=logits)

        sequence_length = self._get_sequence_length()
        sequence_weights = tf.convert_to_tensor(weights)  # _get_weights_and_check_match_logits may return scalar
        sequence_weights = make_sequence_weights(sequence_weights, logits, sequence_length)

        masked_loss = mask_real_sequence(unweighted_loss, sequence_length)
        training_loss = tf.losses.compute_weighted_loss(
            masked_loss, weights=sequence_weights, reduction=self._loss_reduction)

        return head.LossSpec(
            training_loss=training_loss,
            unreduced_loss=masked_loss,
            weights=sequence_weights,
            processed_labels=labels)


class _MultiClassHeadWithSoftmaxCrossEntropyLoss(
    head._MultiClassHeadWithSoftmaxCrossEntropyLoss, SequenceLengthProvider):
    def create_loss(self, features, mode, logits, labels):
        """See `Head`."""
        del mode  # Unused for this head.

        logits = tf.convert_to_tensor(logits)
        logits.get_shape().assert_has_rank(3)

        labels = head._check_dense_labels_match_logits_and_reshape(
            labels=labels, logits=logits, expected_labels_dimension=1)
        labels.get_shape().assert_has_rank(3)
        label_ids = self._label_ids(labels)

        if self._loss_fn:
            unweighted_loss = head._call_loss_fn(
                loss_fn=self._loss_fn, labels=label_ids, logits=logits, features=features, expected_loss_dim=1)
        else:
            unweighted_loss = tf.losses.sparse_softmax_cross_entropy(
                labels=label_ids, logits=logits, reduction=tf.losses.Reduction.NONE)
            # Restore the squeezed dim, so unweighted_loss matches the weights shape.
            unweighted_loss = tf.expand_dims(unweighted_loss, axis=-1)

        weights = head._get_weights_and_check_match_logits(
            features=features, weight_column=self._weight_column, logits=logits)

        sequence_length = self._get_sequence_length()
        sequence_weights = tf.convert_to_tensor(weights)  # _get_weights_and_check_match_logits may return scalar
        sequence_weights = make_sequence_weights(sequence_weights, logits, sequence_length)

        masked_loss = mask_real_sequence(unweighted_loss, sequence_length)
        training_loss = tf.losses.compute_weighted_loss(
            masked_loss, weights=sequence_weights, reduction=self._loss_reduction)

        return head.LossSpec(
            training_loss=training_loss,
            unreduced_loss=masked_loss,
            weights=sequence_weights,
            processed_labels=label_ids)


class _SingleClassHeadWithCrfLogLikelihoodLoss(
    head._MultiClassHeadWithSoftmaxCrossEntropyLoss, SequenceLengthProvider):
    """See `sequence_multi_class_head_with_crf_loss`."""

    def __init__(self, *args, **kwargs):
        super(_SingleClassHeadWithCrfLogLikelihoodLoss, self).__init__(*args, **kwargs)
        self._transition_params = tf.get_variable(
            "transition_params", shape=[self.logits_dimension, self.logits_dimension], dtype=tf.float32)

    def create_loss(self, features, mode, logits, labels):
        """See `Head`."""

        del mode  # Unused for this head.

        logits = tf.convert_to_tensor(logits)
        logits.get_shape().assert_has_rank(3)

        labels = head._check_dense_labels_match_logits_and_reshape(
            labels=labels, logits=logits, expected_labels_dimension=1)
        labels = tf.squeeze(labels, axis=-1)
        labels.get_shape().assert_has_rank(2)
        label_ids = self._label_ids(labels)

        sequence_length = self._get_sequence_length()

        # Compute the log-likelihood of sequences and keep the transition params for inference.
        log_likelihood, _ = crf.crf_log_likelihood(
            inputs=logits,
            tag_indices=label_ids,
            sequence_lengths=sequence_length,
            transition_params=self._transition_params)
        neg_log_likelihood = -log_likelihood

        # Obtain user-defined weights
        user_weights = head._get_weights_and_check_match_logits(
            features=features, weight_column=self._weight_column, logits=logits)
        user_weights = tf.convert_to_tensor(user_weights)  # _get_weights_and_check_match_logits may return scalar

        # To avoid unnecessary loss computations, we should expand and then reduce user weights to same shape as
        # neg_log_likelihood: [batch_size]
        weights = tf.multiply(user_weights, tf.ones_like(logits, dtype=tf.float32))  # now rank should be equal 3
        weights = tf.reduce_mean(tf.reduce_mean(weights, axis=-1), axis=-1)
        weights.get_shape().assert_has_rank(1)

        training_loss = tf.losses.compute_weighted_loss(
            neg_log_likelihood, weights=weights, reduction=self._loss_reduction)

        # Restore the squeezed dims, so unweighted_loss matches the labels shape.
        _, max_length, _ = tf.unstack(tf.shape(logits))
        unweighted_loss = tf.divide(neg_log_likelihood, tf.to_float(sequence_length))
        unweighted_loss = tf.expand_dims(tf.expand_dims(unweighted_loss, axis=-1), axis=-1)
        unweighted_loss = tf.tile(unweighted_loss, [1, max_length, 1])
        unweighted_loss = mask_real_sequence(unweighted_loss, sequence_length)

        # Compute weights for metrics evaluation
        metrics_weights = make_sequence_weights(user_weights, logits, sequence_length)

        return head.LossSpec(
            training_loss=training_loss,
            unreduced_loss=unweighted_loss,
            weights=metrics_weights,
            processed_labels=label_ids)

    def create_estimator_spec(
            self, features, mode, logits, labels=None, optimizer=None, train_op_fn=None, regularization_losses=None):
        """Returns an `EstimatorSpec`.

        Args:
          features: Input `dict` of `Tensor` or `SparseTensor` objects.
          mode: Estimator's `ModeKeys`.
          logits: `Tensor` with shape `[batch_size, max_time, logits_dimension]`.
          labels: integer or string `Tensor` with shape `[batch_size, max_time]`. `labels` is required argument when `mode`
            equals `TRAIN` or `EVAL`.
          optimizer: `Optimizer` instance to optimize the loss in TRAIN mode. Namely, sets
            `train_op = optimizer.minimize(loss, global_step)`, which updates variables and increments `global_step`.
          train_op_fn: Function that takes a scalar loss `Tensor` and returns `train_op`. Used if `optimizer` is `None`.
          regularization_losses: A list of additional scalar tf.losses to be added to the training loss, such as
            regularization tf.losses. These tf.losses are usually expressed as a batch average, so for best results
            users need to set `loss_reduction=SUM_OVER_BATCH_SIZE` or `loss_reduction=SUM_OVER_NONZERO_WEIGHTS` when
            creating the head to avoid scaling errors.
        Returns:
          `EstimatorSpec`.
        """

        logits = tf.convert_to_tensor(logits)
        logits.get_shape().assert_has_rank(3)
        if labels is not None:
            labels = tf.convert_to_tensor(labels)

        with tf.name_scope(self._name, 'head'):
            logits = head._check_logits_final_dim(logits, self.logits_dimension)

            # Predict.
            pred_keys = prediction_keys.PredictionKeys
            with tf.name_scope(None, 'predictions', (logits,)):
                class_ids, _ = crf.crf_decode(
                    potentials=logits,
                    transition_params=self._transition_params,
                    sequence_length=self._get_sequence_length())
                class_ids = tf.to_int64(class_ids)

                # class_ids's shape is [D0, D1, ... DN].
                class_ids = tf.expand_dims(class_ids, axis=-1)
                if self._label_vocabulary:
                    table = lookup_ops.index_to_string_table_from_tensor(
                        vocabulary_list=self._label_vocabulary, name='class_string_lookup')
                    classes = table.lookup(class_ids)
                else:
                    classes = tf.as_string(class_ids, name='str_classes')

                probabilities = tf.nn.softmax(logits, name=pred_keys.PROBABILITIES)
                predictions = {
                    pred_keys.LOGITS: logits,
                    pred_keys.PROBABILITIES: probabilities,
                    pred_keys.CLASS_IDS: class_ids,
                    pred_keys.CLASSES: classes,
                }
            if mode == tf.estimator.ModeKeys.PREDICT:
                # classifier_output = _sequence_classification_output(
                classifier_output = head._classification_output(
                    scores=probabilities, n_classes=self._n_classes, label_vocabulary=self._label_vocabulary)
                return tf.estimator.EstimatorSpec(
                    mode=tf.estimator.ModeKeys.PREDICT,
                    predictions=predictions,
                    export_outputs={
                        head._DEFAULT_SERVING_KEY: classifier_output,
                        head._CLASSIFY_SERVING_KEY: classifier_output,
                        head._PREDICT_SERVING_KEY: export_output.PredictOutput(predictions)
                    })

            training_loss, unreduced_loss, weights, label_ids = self.create_loss(
                features=features, mode=mode, logits=logits, labels=labels)
            if regularization_losses:
                regularization_loss = tf.add_n(regularization_losses)
                regularized_training_loss = tf.add_n([training_loss, regularization_loss])
            else:
                regularization_loss = None
                regularized_training_loss = training_loss
            # Eval.
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode=tf.estimator.ModeKeys.EVAL,
                    predictions=predictions,
                    loss=regularized_training_loss,
                    eval_metric_ops=self._eval_metric_ops(
                        labels=label_ids,
                        class_ids=class_ids,
                        weights=weights,
                        unreduced_loss=unreduced_loss,
                        regularization_loss=regularization_loss))

            # Train.
            if optimizer is not None:
                if train_op_fn is not None:
                    raise ValueError('train_op_fn and optimizer cannot both be set.')
                train_op = optimizer.minimize(regularized_training_loss, global_step=tf.train.get_global_step())
            elif train_op_fn is not None:
                train_op = train_op_fn(regularized_training_loss)
            else:
                raise ValueError('train_op_fn and optimizer cannot both be None.')

            # Only summarize mean_loss for SUM reduction to preserve backwards compatibility. Otherwise skip it
            # to avoid unnecessary computation.
            if self._loss_reduction == tf.losses.Reduction.SUM:
                example_weight_sum = tf.reduce_sum(weights * tf.ones_like(unreduced_loss))
                mean_loss = training_loss / example_weight_sum
            else:
                mean_loss = None

        with tf.name_scope(''):
            keys = metric_keys.MetricKeys
            tf.summary.scalar(head._summary_key(self._name, keys.LOSS), regularized_training_loss)
            if mean_loss is not None:
                tf.summary.scalar(head._summary_key(self._name, keys.LOSS_MEAN), mean_loss)
            if regularization_loss is not None:
                tf.summary.scalar(head._summary_key(self._name, keys.LOSS_REGULARIZATION), regularization_loss)

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            predictions=predictions,
            loss=regularized_training_loss,
            train_op=train_op)


class _MultiHead(multi_head._MultiHead, SequenceLengthProvider):
    def set_sequence_length(self, actual_length):
        super(_MultiHead, self).set_sequence_length(actual_length)

        for head in self._heads:
            head.set_sequence_length(actual_length)

# def _sequence_classification_output(scores, n_classes, label_vocabulary=None):
#     """Create classification output for sequential result.
#     Based on head._classification_output
#     """
#
#     batch_size, max_len, _ = tf.unstack(tf.shape(scores))
#     if label_vocabulary:
#         export_class_list = label_vocabulary
#     else:
#         export_class_list = tf.as_string(tf.range(n_classes))
#     export_output_classes = tf.tile(
#         input=tf.expand_dims(input=export_class_list, axis=0),
#         multiples=[batch_size * max_len, 1])
#     export_output_classes = tf.reshape(export_output_classes, [batch_size, max_len, -1])
#
#     return export_output.ClassificationOutput(
#         scores=scores,
#         # `ClassificationOutput` requires string classes.
#         classes=export_output_classes)
