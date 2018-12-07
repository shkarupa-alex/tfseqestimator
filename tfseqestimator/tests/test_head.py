from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.training import monitored_session
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.saved_model import signature_constants
from ..head import sequence_multi_class_head_with_crf_loss, sequence_regression_head_with_mse_loss
from ..head import sequence_binary_classification_head_with_sigmoid, sequence_multi_class_head_with_softmax
from ..head import sequence_multi_head

_DEFAULT_SERVING_KEY = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY


def _initialize_variables(test_case, scaffold):
    scaffold.finalize()
    test_case.assertIsNone(scaffold.init_feed_dict)
    test_case.assertIsNone(scaffold.init_fn)
    scaffold.init_op.run()
    scaffold.ready_for_local_init_op.eval()
    scaffold.local_init_op.run()
    scaffold.ready_op.eval()
    test_case.assertIsNotNone(scaffold.saver)


def _assert_simple_summaries(test_case, expected_summaries, summary_str,
                             tol=1e-6):
    """Assert summary the specified simple values.
    Args:
      test_case: test case.
      expected_summaries: Dict of expected tags and simple values.
      summary_str: Serialized `summary_pb2.Summary`.
      tol: Tolerance for relative and absolute.
    """
    summary = summary_pb2.Summary()
    summary.ParseFromString(summary_str)
    test_case.assertAllClose(expected_summaries, {v.tag: v.simple_value for v in summary.value}, rtol=tol, atol=tol)


def _assert_no_hooks(test_case, spec):
    test_case.assertAllEqual([], spec.training_chief_hooks)
    test_case.assertAllEqual([], spec.training_hooks)


class MultiClassHeadWithCrfLogLikelihoodLossTest(tf.test.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        tf.set_random_seed(1)

    def test_n_classes_is_none(self):
        with self.assertRaisesRegexp(ValueError, 'n_classes must be > 2'):
            sequence_multi_class_head_with_crf_loss(n_classes=None)

    def test_n_classes_is_2(self):
        with self.assertRaisesRegexp(ValueError, 'n_classes must be > 2'):
            sequence_multi_class_head_with_crf_loss(n_classes=2)

    def test_invalid_loss_reduction(self):
        with self.assertRaisesRegexp(ValueError, r'Invalid loss_reduction: invalid_loss_reduction'):
            sequence_multi_class_head_with_crf_loss(n_classes=3, loss_reduction='invalid_loss_reduction')
        with self.assertRaisesRegexp(ValueError, r'Invalid loss_reduction: none'):
            sequence_multi_class_head_with_crf_loss(n_classes=3, loss_reduction=tf.losses.Reduction.NONE)

    def test_invalid_logits_shape(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes)
        head.set_sequence_length([1, 1])
        self.assertEqual(n_classes, head.logits_dimension)

        # Logits should be shape (batch_size, max_len, 3).
        logits_2x1x2 = np.array((((45., 44.),), ((41., 42.),)))

        # Static shape.
        with self.assertRaisesRegexp(ValueError, 'logits shape'):
            head.create_estimator_spec(
                features={'x': np.array(((30.,), (42.,),))},
                mode=tf.estimator.ModeKeys.PREDICT,
                logits=logits_2x1x2)

        # Dynamic shape.
        logits_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, None, 2))
        with self.assertRaisesRegexp(ValueError, 'logits shape'):
            head.create_estimator_spec(
                features={'x': np.array(((30.,), (42.,),))},
                mode=tf.estimator.ModeKeys.PREDICT,
                logits=logits_placeholder)

    def test_invalid_labels_shape(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes)
        head.set_sequence_length([1, 1])
        self.assertEqual(n_classes, head.logits_dimension)

        # Logits should be shape (batch_size, max_len, 3).
        # Labels should be shape (batch_size, max_len, 1).
        labels_2x1x2 = np.array((((45, 44),), ((41, 42),)), dtype=np.int)
        logits_2x1x3 = np.array((((1., 2., 3.),), ((1., 2., 3.),)))
        features = {'x': np.array((((42.,),)))}

        # Static shape.
        with self.assertRaisesRegexp(ValueError, 'Mismatched label shape'):
            head.create_loss(features=features,
                             mode=tf.estimator.ModeKeys.EVAL,
                             logits=logits_2x1x3,
                             labels=labels_2x1x2)

        # Dynamic shape.
        labels_placeholder = tf.placeholder(dtype=tf.int64)
        logits_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, None, 3))
        training_loss = head.create_loss(
            features=features,
            mode=tf.estimator.ModeKeys.EVAL,
            logits=logits_placeholder,
            labels=labels_placeholder)[0]
        with self.test_session():
            tf.global_variables_initializer().run()
            with self.assertRaisesRegexp(
                    tf.errors.InvalidArgumentError,
                    r'\[expected_labels_shape: \] \[2 1 1\] \[labels_shape: \] \[2 1 2\]'):
                training_loss.eval({
                    logits_placeholder: logits_2x1x3,
                    labels_placeholder: labels_2x1x2
                })

    def test_invalid_labels_type(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes)
        head.set_sequence_length([1, 1])
        self.assertEqual(n_classes, head.logits_dimension)

        # Logits should be shape (batch_size, 1, 3).
        # Labels should be shape (batch_size, 1, 1).
        labels_2x1x1 = np.array((((1.,),), ((1.,),)))
        logits_2x1x3 = np.array((((1., 2., 3.),), ((1., 2., 3.),)))
        features = {'x': np.array((((42.,),)))}

        # Static shape.
        with self.assertRaisesRegexp(ValueError, 'Labels dtype'):
            head.create_loss(features=features,
                             mode=tf.estimator.ModeKeys.EVAL,
                             logits=logits_2x1x3,
                             labels=labels_2x1x1)

        # Dynamic shape.
        labels_placeholder = tf.placeholder(dtype=tf.float32)
        logits_placeholder = tf.placeholder(dtype=tf.float32)
        with self.assertRaisesRegexp(ValueError, 'Labels dtype'):
            head.create_loss(features=features,
                             mode=tf.estimator.ModeKeys.EVAL,
                             logits=logits_placeholder,
                             labels=labels_placeholder)

    def test_invalid_labels_values(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes)
        head.set_sequence_length([1, 1])
        self.assertEqual(n_classes, head.logits_dimension)

        labels_2x1x1_with_large_id = np.array((((45,),), ((1,),)), dtype=np.int)
        labels_2x1x1_with_negative_id = np.array((((-5,),), ((1,),)), dtype=np.int)
        logits_2x1x3 = np.array((((1., 2., 4.),), ((1., 2., 3.),)))

        labels_placeholder = tf.placeholder(dtype=tf.int64)
        logits_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, None, 3))
        training_loss = head.create_loss(features={'x': np.array((((42.,),),))},
                                         mode=tf.estimator.ModeKeys.EVAL,
                                         logits=logits_placeholder,
                                         labels=labels_placeholder)[0]
        with self.test_session() as sess:
            tf.global_variables_initializer().run()
            with self.assertRaisesOpError('Labels must <= n_classes - 1'):
                training_loss.eval({
                    labels_placeholder: labels_2x1x1_with_large_id,
                    logits_placeholder: logits_2x1x3
                })

        with self.test_session():
            tf.global_variables_initializer().run()
            with self.assertRaisesOpError('Labels must >= 0'):
                training_loss.eval({
                    labels_placeholder: labels_2x1x1_with_negative_id,
                    logits_placeholder: logits_2x1x3
                })

    def test_invalid_labels_sparse_tensor(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes)
        head.set_sequence_length([1, 1])
        self.assertEqual(n_classes, head.logits_dimension)

        labels_2x1x1 = tf.SparseTensor(
            values=['english', 'italian'],
            indices=[[0, 0, 0], [1, 0, 0]],
            dense_shape=[2, 1, 1])
        logits_2x1x3 = np.array((((1., 2., 4.),), ((1., 2., 3.),)))

        with self.assertRaisesRegexp(ValueError, 'SparseTensor labels are not supported.'):
            head.create_loss(features={'x': np.array(((42.,),))},
                             mode=tf.estimator.ModeKeys.EVAL,
                             logits=logits_2x1x3,
                             labels=labels_2x1x1)

    def test_incompatible_labels_shape(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes)
        head.set_sequence_length([1, 1])
        self.assertEqual(n_classes, head.logits_dimension)

        # Logits should be shape (batch_size, 1, 3).
        # Labels should be shape (batch_size, 1, 1).
        # Here batch sizes are different.
        labels_3x1x1 = np.array((((1,),), ((1,),), ((1,),),))
        values_2x1x3 = np.array((((1., 2., 3.),), ((1., 2., 3.),)))
        features = {'x': values_2x1x3}

        # Static shape.
        with self.assertRaisesRegexp(
                ValueError, 'Dimension 0 in both shapes must be equal, but are 2 and 3'):
            head.create_loss(features=features,
                             mode=tf.estimator.ModeKeys.EVAL,
                             logits=values_2x1x3,
                             labels=labels_3x1x1)

        # Dynamic shape.
        labels_placeholder = tf.placeholder(dtype=tf.int64)
        logits_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, None, 3))
        training_loss = head.create_loss(features=features,
                                         mode=tf.estimator.ModeKeys.EVAL,
                                         logits=logits_placeholder,
                                         labels=labels_placeholder)[0]
        with self.test_session() as sess:
            tf.global_variables_initializer().run()
            with self.assertRaisesRegexp(
                    tf.errors.InvalidArgumentError,
                    r'\[expected_labels_shape: \] \[2 1 1\] \[labels_shape: \] \[3 1 1\]'):
                training_loss.eval({
                    labels_placeholder: labels_3x1x1,
                    logits_placeholder: values_2x1x3
                })

    def test_name(self):
        head = sequence_multi_class_head_with_crf_loss(n_classes=3, name='foo')
        self.assertEqual('foo', head.name)

    def test_predict(self):
        # tf.set_random_seed(1)
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes)
        head.set_sequence_length([2])
        self.assertEqual(n_classes, head.logits_dimension)

        logits = [[[1., 0., 0.], [0., 0., 1.]]]
        expected_probabilities = [[[0.576117, 0.2119416, 0.2119416], [0.2119416, 0.2119416, 0.576117]]]
        expected_class_ids = [[[0], [2]]]
        expected_classes = [[[b'0'], [b'2']]]
        # expected_export_classes = [[[b'0', b'1', b'2'], [b'0', b'1', b'2']]]
        expected_export_classes = [[b'0', b'1', b'2']]  # TODO: tile

        spec = head.create_estimator_spec(
            features={'x': np.array((((42,),),), dtype=np.int32)},
            mode=tf.estimator.ModeKeys.PREDICT,
            logits=logits)

        self.assertItemsEqual(
            (_DEFAULT_SERVING_KEY, 'predict', 'classification'),
            spec.export_outputs.keys())

        # Assert predictions and export_outputs.
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertIsNone(spec.scaffold.summary_op)
            predictions = sess.run(spec.predictions)
            self.assertAllClose(logits, predictions[prediction_keys.PredictionKeys.LOGITS])
            self.assertAllClose(expected_probabilities, predictions[prediction_keys.PredictionKeys.PROBABILITIES])
            self.assertAllClose(expected_class_ids, predictions[prediction_keys.PredictionKeys.CLASS_IDS])
            self.assertAllEqual(expected_classes, predictions[prediction_keys.PredictionKeys.CLASSES])

            self.assertAllClose(expected_probabilities, sess.run(spec.export_outputs[_DEFAULT_SERVING_KEY].scores))
            self.assertAllEqual(expected_export_classes, sess.run(spec.export_outputs[_DEFAULT_SERVING_KEY].classes))

    def test_predict_with_vocabulary_list(self):
        # tf.set_random_seed(1)
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes, label_vocabulary=['aang', 'iroh', 'zuko'])
        head.set_sequence_length([2])

        logits = [[[1., 0., 0.], [0., 0., 1.]]]
        expected_classes = [[[b'aang'], [b'zuko']]]
        # expected_export_classes = [[[b'aang', b'iroh', b'zuko'], [b'aang', b'iroh', b'zuko']]]
        expected_export_classes = [[b'aang', b'iroh', b'zuko']]  # TODO: tile

        spec = head.create_estimator_spec(
            features={'x': np.array((((42,),),), dtype=np.int32)},
            mode=tf.estimator.ModeKeys.PREDICT,
            logits=logits)

        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertAllEqual(expected_classes, sess.run(spec.predictions[prediction_keys.PredictionKeys.CLASSES]))
            self.assertAllEqual(expected_export_classes, sess.run(spec.export_outputs[_DEFAULT_SERVING_KEY].classes))

    def test_weight_should_not_impact_prediction(self):
        n_classes = 3
        logits = [[[1., 0., 0.], [0., 0., 1.]]]
        expected_probabilities = [[[0.576117, 0.2119416, 0.2119416],
                                   [0.2119416, 0.2119416, 0.576117]]]
        head = sequence_multi_class_head_with_crf_loss(n_classes, weight_column='label_weights')
        head.set_sequence_length([2])

        weights_1x2x1 = [[[1.], [2.]]]
        spec = head.create_estimator_spec(
            features={
                'x': np.array((((42,),),), dtype=np.int32),
                'label_weights': weights_1x2x1,
            },
            mode=tf.estimator.ModeKeys.PREDICT,
            logits=logits)

        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            predictions = sess.run(spec.predictions)
            self.assertAllClose(logits, predictions[prediction_keys.PredictionKeys.LOGITS])
            self.assertAllClose(expected_probabilities, predictions[prediction_keys.PredictionKeys.PROBABILITIES])

    def test_eval_create_loss(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes)
        head.set_sequence_length([2])

        logits = np.array((((10, 0, 0), (0, 10, 0),),), dtype=np.float32)
        labels = np.array((((1,), (1,)),), dtype=np.int64)
        features = {'x': np.array((((42,),),), dtype=np.int32)}
        # loss = cross_entropy(labels, logits) = [10, 0].
        expected_training_loss = 10.1
        # Create loss.
        training_loss = head.create_loss(features=features,
                                         mode=tf.estimator.ModeKeys.EVAL,
                                         logits=logits,
                                         labels=labels)[0]
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            self.assertAllClose(expected_training_loss, training_loss.eval(), rtol=1e-2, atol=1e-2)

    def test_eval_labels_none(self):
        # Tests that error is raised when labels is None.
        head = sequence_multi_class_head_with_crf_loss(n_classes=3)
        head.set_sequence_length([2])

        with self.assertRaisesRegexp(ValueError, r'You must provide a labels Tensor\. Given: None\.'):
            head.create_estimator_spec(
                features={'x': np.array((((42,),),), dtype=np.int32)},
                mode=tf.estimator.ModeKeys.EVAL,
                logits=np.array((((10, 0, 0), (0, 10, 0),),), dtype=np.float32),
                labels=None)

    def test_eval(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes)
        head.set_sequence_length([2])
        logits = np.array((((10, 0, 0), (0, 10, 0),),), dtype=np.float32)
        labels = np.array((((1,), (1,)),), dtype=np.int64)
        features = {'x': np.array((((42,),),), dtype=np.int32)}
        # loss = sum(cross_entropy(labels, logits)) = sum(10, 0) = 10.
        expected_loss = 10.1
        # Create estimator spec.
        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.EVAL,
            logits=logits,
            labels=labels)

        keys = metric_keys.MetricKeys
        expected_metrics = {
            keys.LOSS_MEAN: expected_loss / 2,
            keys.ACCURACY: 0.5,  # 1 of 2 labels is correct.
        }

        # Assert spec contains expected tensors.
        self.assertIsNotNone(spec.loss)
        self.assertItemsEqual(expected_metrics.keys(), spec.eval_metric_ops.keys())
        self.assertIsNone(spec.train_op)
        self.assertIsNone(spec.export_outputs)
        _assert_no_hooks(self, spec)

        # Assert predictions, loss, and metrics.
        tol = 1e-2
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertIsNone(spec.scaffold.summary_op)
            value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
            update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
            loss, metrics = sess.run((spec.loss, update_ops))
            self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
            # Check results of both update (in `metrics`) and value ops.
            self.assertAllClose(expected_metrics, metrics, rtol=tol, atol=tol)
            self.assertAllClose(expected_metrics, {k: value_ops[k].eval() for k in value_ops}, rtol=tol, atol=tol)

    def test_eval_metric_ops_with_head_name(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes, name='some_multiclass_head')
        head.set_sequence_length([2])
        logits = np.array((((10, 0, 0), (0, 10, 0),),), dtype=np.float32)
        labels = np.array((((1,), (1,)),), dtype=np.int64)
        features = {'x': np.array((((42,),),), dtype=np.int32)}
        # Create estimator spec.
        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.EVAL,
            logits=logits,
            labels=labels)

        expected_metric_keys = [
            '{}/some_multiclass_head'.format(metric_keys.MetricKeys.LOSS_MEAN),
            '{}/some_multiclass_head'.format(metric_keys.MetricKeys.ACCURACY)
        ]
        self.assertItemsEqual(expected_metric_keys, spec.eval_metric_ops.keys())

    def test_eval_with_regularization_losses(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(
            n_classes, loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        head.set_sequence_length([2])
        logits = np.array((((10, 0, 0), (0, 10, 0),),), dtype=np.float32)
        labels = np.array((((1,), (1,)),), dtype=np.int64)
        features = {'x': np.array((((42,),),), dtype=np.int32)}
        regularization_losses = [1.5, 0.5]
        expected_regularization_loss = 2.
        # unregularized_loss = sum(cross_entropy(labels, logits)) / batch_size
        #                    = sum(10, 0) / 2 = 5.
        expected_unregularized_loss = 10.1
        expected_regularized_loss = (expected_unregularized_loss + expected_regularization_loss)
        # Create estimator spec.
        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.EVAL,
            logits=logits,
            labels=labels,
            regularization_losses=regularization_losses)

        keys = metric_keys.MetricKeys
        expected_metrics = {
            keys.LOSS_MEAN: expected_unregularized_loss / 2,
            keys.LOSS_REGULARIZATION: expected_regularization_loss,
            keys.ACCURACY: 0.5,  # 1 of 2 labels is correct.
        }

        # Assert predictions, loss, and metrics.
        tol = 1e-2
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertIsNone(spec.scaffold.summary_op)
            value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
            update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
            loss, metrics = sess.run((spec.loss, update_ops))
            self.assertAllClose(expected_regularized_loss, loss, rtol=tol, atol=tol)
            # Check results of both update (in `metrics`) and value ops.
            self.assertAllClose(expected_metrics, metrics, rtol=tol, atol=tol)
            self.assertAllClose(expected_metrics, {k: value_ops[k].eval() for k in value_ops}, rtol=tol, atol=tol)

    def test_eval_with_label_vocabulary_create_loss(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes, label_vocabulary=['aang', 'iroh', 'zuko'])
        head.set_sequence_length([2])
        logits = [[[10., 0, 0], [0, 10, 0]]]
        labels = [[[b'iroh'], [b'iroh']]]
        features = {'x': np.array((((42,),),), dtype=np.int32)}
        # loss = cross_entropy(labels, logits) = [10, 0].
        expected_training_loss = 10.1
        training_loss = head.create_loss(features=features,
                                         mode=tf.estimator.ModeKeys.EVAL,
                                         logits=logits,
                                         labels=labels)[0]
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            self.assertAllClose(expected_training_loss, training_loss.eval(), rtol=1e-2, atol=1e-2)

    def test_eval_with_label_vocabulary(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes, label_vocabulary=['aang', 'iroh', 'zuko'])
        head.set_sequence_length([2])

        logits = [[[10., 0, 0], [0, 10, 0]]]
        labels = [[[b'iroh'], [b'iroh']]]
        features = {'x': np.array((((42,),),), dtype=np.int32)}
        # loss = sum(cross_entropy(labels, logits)) = sum(10, 0) = 10.
        expected_loss = 10.1
        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.EVAL,
            logits=logits,
            labels=labels)

        keys = metric_keys.MetricKeys
        expected_metrics = {
            keys.LOSS_MEAN: expected_loss / 2,
            keys.ACCURACY: 0.5,  # 1 of 2 labels is correct.
        }

        tol = 1e-2
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
            update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
            loss, metrics = sess.run((spec.loss, update_ops))
            self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
            # Check results of both update (in `metrics`) and value ops.
            self.assertAllClose(expected_metrics, metrics, rtol=tol, atol=tol)
            self.assertAllClose(expected_metrics, {k: value_ops[k].eval() for k in value_ops}, rtol=tol, atol=tol)

    def test_weighted_multi_example_eval(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes, weight_column='label_weights')
        head.set_sequence_length([1, 1, 1])

        # Create estimator spec.
        logits = np.array((((10, 0, 0),), ((0, 10, 0),), ((0, 0, 10),),), dtype=np.float32)
        labels = np.array((((1,),), ((2,),), ((2,),)), dtype=np.int64)
        weights_3x1 = np.array(((1.,), (2.,), (3.,)), dtype=np.float64)
        # loss = sum(cross_entropy(labels, logits) * [1, 2, 3])
        #      = sum([10, 10, 0] * [1, 2, 3]) = 30
        expected_loss = 30.3
        spec = head.create_estimator_spec(
            features={
                'x': np.array((((42,),),), dtype=np.int32),
                'label_weights': weights_3x1,
            },
            mode=tf.estimator.ModeKeys.EVAL,
            logits=logits,
            labels=labels)

        keys = metric_keys.MetricKeys
        expected_metrics = {
            keys.LOSS_MEAN: expected_loss / np.sum(weights_3x1),
            # Weighted accuracy is 1 * 3.0 / sum weights = 0.5
            keys.ACCURACY: 0.5,
        }

        # Assert spec contains expected tensors.
        self.assertIsNotNone(spec.loss)
        self.assertItemsEqual(expected_metrics.keys(), spec.eval_metric_ops.keys())
        self.assertIsNone(spec.train_op)
        self.assertIsNone(spec.export_outputs)
        _assert_no_hooks(self, spec)

        # Assert loss, and metrics.
        tol = 1e-2
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertIsNone(spec.scaffold.summary_op)
            value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
            update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
            loss, metrics = sess.run((spec.loss, update_ops))
            self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
            # Check results of both update (in `metrics`) and value ops.
            self.assertAllClose(expected_metrics, metrics, rtol=tol, atol=tol)
            self.assertAllClose(expected_metrics, {k: value_ops[k].eval() for k in value_ops}, rtol=tol, atol=tol)

    def test_train_create_loss(self):
        head = sequence_multi_class_head_with_crf_loss(n_classes=3)
        head.set_sequence_length([1, 1])

        logits = np.array((((10, 0, 0),), ((0, 10, 0),),), dtype=np.float32)
        labels = np.array((((1,),), ((1,),)), dtype=np.int64)
        features = {'x': np.array((((42,),),), dtype=np.int32)}

        # unreduced_loss = cross_entropy(labels, logits) = [10, 0].
        expected_unreduced_loss = [[[10.]], [[0.]]]
        # Weights default to 1.
        expected_weights = [[[1.]], [[1.]]]  # sequence weights
        # training_loss = 1 * 10 + 1 * 0
        expected_training_loss = 10.1
        training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels)
        tol = 1e-2
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            self.assertAllClose(expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)
            self.assertAllClose(expected_unreduced_loss, unreduced_loss.eval(), rtol=tol, atol=tol)
            self.assertAllClose(expected_weights, actual_weights)

    def test_train_create_loss_loss_reduction(self):
        # Tests create_loss with loss_reduction.
        head = sequence_multi_class_head_with_crf_loss(
            n_classes=3, loss_reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        head.set_sequence_length([1, 1])

        logits = np.array((((10, 0, 0),), ((0, 10, 0),),), dtype=np.float32)
        labels = np.array((((1,),), ((1,),)), dtype=np.int64)
        features = {'x': np.array((((42,),),), dtype=np.int32)}

        # unreduced_loss = cross_entropy(labels, logits) = [10, 0].
        expected_unreduced_loss = [[[10.]], [[0.]]]
        # Weights default to 1.
        expected_weights = [[[1.]], [[1.]]]  # sequence weights
        # training_loss = 1 * 10 + 1 * 0 / num_nonzero_weights
        expected_training_loss = 10.1 / 2.
        training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels)
        tol = 1e-2
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            self.assertAllClose(expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)
            self.assertAllClose(expected_unreduced_loss, unreduced_loss.eval(), rtol=tol, atol=tol)
            self.assertAllClose(expected_weights, actual_weights)

    def test_train_labels_none(self):
        # Tests that error is raised when labels is None.
        head = sequence_multi_class_head_with_crf_loss(n_classes=3)
        head.set_sequence_length([2])

        def _no_op_train_fn(loss):
            del loss
            return tf.no_op()

        with self.assertRaisesRegexp(ValueError, r'You must provide a labels Tensor\. Given: None\.'):
            head.create_estimator_spec(
                features={'x': np.array((((42,),),), dtype=np.int32)},
                mode=tf.estimator.ModeKeys.TRAIN,
                logits=np.array((((10, 0, 0), (0, 10, 0),),), dtype=np.float32),
                labels=None,
                train_op_fn=_no_op_train_fn)

    def test_train(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes)
        head.set_sequence_length([2])

        logits = np.array((((10, 0, 0), (0, 10, 0),),), dtype=np.float32)
        labels = np.array((((1,), (1,)),), dtype=np.int64)
        features = {'x': np.array((((42,),),), dtype=np.int32)}
        expected_train_result = 'my_train_op'

        def _train_op_fn(loss):
            return tf.string_join([tf.constant(expected_train_result), tf.as_string(loss, precision=2)])

        # loss = sum(cross_entropy(labels, logits)) = sum(10, 0) = 10.
        expected_loss = 10.15
        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels,
            train_op_fn=_train_op_fn)

        self.assertIsNotNone(spec.loss)
        self.assertEqual({}, spec.eval_metric_ops)
        self.assertIsNotNone(spec.train_op)
        self.assertIsNone(spec.export_outputs)
        _assert_no_hooks(self, spec)

        # Assert predictions, loss, train_op, and summaries.
        tol = 1e-2
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertIsNotNone(spec.scaffold.summary_op)
            loss, train_result, summary_str = sess.run((spec.loss, spec.train_op, spec.scaffold.summary_op))
            self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
            self.assertEqual(six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)), train_result)
            _assert_simple_summaries(self, {
                metric_keys.MetricKeys.LOSS: expected_loss,
                metric_keys.MetricKeys.LOSS_MEAN: expected_loss / 2,
            }, summary_str, tol)

    def test_train_with_optimizer(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes)
        head.set_sequence_length([2])

        logits = np.array((((10, 0, 0), (0, 10, 0),),), dtype=np.float32)
        labels = np.array((((1,), (1,)),), dtype=np.int64)
        features = {'x': np.array((((42,),),), dtype=np.int32)}
        expected_train_result = 'my_train_op'

        class _Optimizer(object):
            def minimize(self, loss, global_step):
                del global_step
                return tf.string_join([tf.constant(expected_train_result), tf.as_string(loss, precision=2)])

        # loss = sum(cross_entropy(labels, logits)) = sum(10, 0) = 10.
        expected_loss = 10.15
        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels,
            optimizer=_Optimizer())

        tol = 1e-2
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            loss, train_result = sess.run((spec.loss, spec.train_op))
            self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
            self.assertEqual(six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)), train_result)

    def test_train_summaries_with_head_name(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes, name='some_multiclass_head')
        head.set_sequence_length([2])

        logits = np.array((((10, 0, 0), (0, 10, 0),),), dtype=np.float32)
        labels = np.array((((1,), (1,)),), dtype=np.int64)
        # loss = sum(cross_entropy(labels, logits)) = sum(10, 0) = 10.
        expected_loss = 10.15
        features = {'x': np.array((((42,),),), dtype=np.int32)}

        def _train_op_fn(loss):
            del loss
            return tf.no_op()

        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels,
            train_op_fn=_train_op_fn)

        # Assert summaries.
        tol = 1e-2
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertIsNotNone(spec.scaffold.summary_op)
            summary_str = sess.run(spec.scaffold.summary_op)
            _assert_simple_summaries(self, {
                '{}/some_multiclass_head'.format(metric_keys.MetricKeys.LOSS): expected_loss,
                '{}/some_multiclass_head'.format(metric_keys.MetricKeys.LOSS_MEAN): expected_loss / 2,
            }, summary_str, tol)

    def test_train_with_regularization_losses(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(
            n_classes, loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        head.set_sequence_length([1, 1])

        logits = np.array((((10, 0, 0),), ((0, 10, 0),),), dtype=np.float32)
        labels = np.array((((1,),), ((1,),),), dtype=np.int64)
        features = {'x': np.array((((42,),),), dtype=np.int32)}
        expected_train_result = 'my_train_op'

        def _train_op_fn(loss):
            return tf.string_join([tf.constant(expected_train_result), tf.as_string(loss, precision=2)])

        regularization_losses = [1.5, 0.5]
        expected_regularization_loss = 2.
        # unregularized_loss = sum(cross_entropy(labels, logits)) / batch_size
        #                    = sum(10, 0) / 2 = 5.
        # loss = unregularized_loss + regularization_loss = 7.
        expected_loss = 7.
        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels,
            train_op_fn=_train_op_fn,
            regularization_losses=regularization_losses)

        # Assert predictions, loss, train_op, and summaries.
        tol = 1e-2
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertIsNotNone(spec.scaffold.summary_op)
            loss, train_result, summary_str = sess.run((
                spec.loss, spec.train_op, spec.scaffold.summary_op))
            self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
            self.assertEqual(six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)), train_result)
            _assert_simple_summaries(self, {
                metric_keys.MetricKeys.LOSS: expected_loss,
                metric_keys.MetricKeys.LOSS_REGULARIZATION: (
                    expected_regularization_loss),
            }, summary_str, tol)

    def test_train_one_dim_create_loss(self):
        # Tests create_loss with 1D labels and weights (shape [batch_size]).
        head = sequence_multi_class_head_with_crf_loss(n_classes=3, weight_column='label_weights')
        head.set_sequence_length([1, 1, 1])

        logits = np.array((((10, 0, 0),), ((0, 10, 0),), ((0, 0, 10),),), dtype=np.float32)
        labels_rank_2 = np.array(((1,), (2,), (2,),), dtype=np.int64)
        weights_rank_2 = np.array(((1.,), (2.,), (3.,),), dtype=np.float64)
        features = {
            'x': np.array((((42,),),), dtype=np.float32),
            'label_weights': weights_rank_2
        }

        # unreduced_loss = cross_entropy(labels, logits) = [10, 10, 0].
        expected_unreduced_loss = [[[10.]], [[10.]], [[0.]]]
        # weights are reshaped to [3, 1] to match logits.
        expected_weights = [[[1.]], [[2.]], [[3.]]]
        # training_loss = 1 * 10 + 2 * 10 + 3 * 0 = 30.
        expected_training_loss = 30.
        training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels_rank_2)
        tol = 1e-2
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            self.assertAllClose(expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)
            self.assertAllClose(expected_unreduced_loss, unreduced_loss.eval(), rtol=tol, atol=tol)
            self.assertAllClose(expected_weights, actual_weights.eval())

    def test_train_one_dim(self):
        # Tests train with 2D labels and weights (shape [batch_size]).
        head = sequence_multi_class_head_with_crf_loss(n_classes=3, weight_column='label_weights')
        head.set_sequence_length([1, 1, 1])

        logits = np.array((((10, 0, 0),), ((0, 10, 0),), ((0, 0, 10),),), dtype=np.float32)
        labels_rank_2 = np.array(((1,), (2,), (2,),), dtype=np.int64)
        weights_rank_2 = np.array(((1.,), (2.,), (3.,),), dtype=np.float64)

        self.assertEqual((3, 1), labels_rank_2.shape)
        self.assertEqual((3, 1), weights_rank_2.shape)

        expected_train_result = 'my_train_op'

        def _train_op_fn(loss):
            return tf.string_join([tf.constant(expected_train_result), tf.as_string(loss, precision=2)])

        # loss = sum(cross_entropy(labels, logits) * [1, 2, 3]) = sum([10, 10, 0] * [1, 2, 3]) = 30
        expected_loss = 30.

        features = {
            'x': np.array((((42,),),), dtype=np.float32),
            'label_weights': weights_rank_2
        }
        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels_rank_2,
            train_op_fn=_train_op_fn)

        self.assertIsNotNone(spec.loss)
        self.assertEqual({}, spec.eval_metric_ops)
        self.assertIsNotNone(spec.train_op)
        self.assertIsNone(spec.export_outputs)
        _assert_no_hooks(self, spec)

        # Assert predictions, loss, train_op, and summaries.
        tol = 1e-2
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertIsNotNone(spec.scaffold.summary_op)
            loss, train_result, summary_str = sess.run((
                spec.loss, spec.train_op, spec.scaffold.summary_op))
            self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
            self.assertEqual(six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)), train_result)
            _assert_simple_summaries(self, {
                metric_keys.MetricKeys.LOSS: expected_loss,
                metric_keys.MetricKeys.LOSS_MEAN: (expected_loss / np.sum(weights_rank_2)),
            }, summary_str, tol)

    def test_train_with_vocabulary_create_loss(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes, label_vocabulary=['aang', 'iroh', 'zuko'])
        head.set_sequence_length([2])

        logits = [[[10., 0, 0], [0, 10, 0]]]
        labels = [[[b'iroh'], [b'iroh']]]
        features = {'x': np.array((((42,),),), dtype=np.int32)}
        # loss = cross_entropy(labels, logits) = [10, 0].
        expected_training_loss = 10.14
        training_loss = head.create_loss(features=features,
                                         mode=tf.estimator.ModeKeys.TRAIN,
                                         logits=logits,
                                         labels=labels)[0]
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            self.assertAllClose(expected_training_loss, training_loss.eval(), rtol=1e-2, atol=1e-2)

    def test_train_with_vocabulary(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes, label_vocabulary=['aang', 'iroh', 'zuko'])
        head.set_sequence_length([2])

        logits = [[[10., 0, 0], [0, 10, 0]]]
        labels = [[[b'iroh'], [b'iroh']]]
        features = {'x': np.array((((42,),),), dtype=np.int32)}

        def _train_op_fn(loss):
            del loss
            return tf.no_op()

        # loss = sum(cross_entropy(labels, logits)) = sum(10, 0) = 10.
        expected_loss = 10.14
        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels,
            train_op_fn=_train_op_fn)

        tol = 1e-2
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            loss = sess.run(spec.loss)
            self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)

    def test_weighted_multi_example_train(self):
        n_classes = 3
        head = sequence_multi_class_head_with_crf_loss(n_classes, weight_column='label_weights')
        head.set_sequence_length([1, 1, 1])

        # Create estimator spec.
        logits = np.array((((10, 0, 0),), ((0, 10, 0),), ((0, 0, 10),),), dtype=np.float32)
        labels = np.array((((1,),), ((2,),), ((2,),)), dtype=np.int64)
        weights_3x1x1 = np.array((((1.,),), ((2.,),), ((3.,),)), dtype=np.float64)
        expected_train_result = 'my_train_op'
        # loss = sum(cross_entropy(labels, logits) * [1, 2, 3])
        #      = sum([10, 10, 0] * [1, 2, 3]) = 30
        expected_loss = 30.

        def _train_op_fn(loss):
            return tf.string_join(
                [tf.constant(expected_train_result),
                 tf.as_string(loss, precision=2)])

        spec = head.create_estimator_spec(
            features={
                'x': np.array((((42,),),), dtype=np.float32),
                'label_weights': weights_3x1x1,
            },
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels,
            train_op_fn=_train_op_fn)

        self.assertIsNotNone(spec.loss)
        self.assertEqual({}, spec.eval_metric_ops)
        self.assertIsNotNone(spec.train_op)
        self.assertIsNone(spec.export_outputs)
        _assert_no_hooks(self, spec)

        # Assert predictions, loss, train_op, and summaries.
        tol = 1e-2
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertIsNotNone(spec.scaffold.summary_op)
            loss, train_result, summary_str = sess.run((spec.loss, spec.train_op,
                                                        spec.scaffold.summary_op))
            self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
            self.assertEqual(six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)), train_result)
            _assert_simple_summaries(self, {
                metric_keys.MetricKeys.LOSS: expected_loss,
                # loss mean = sum(cross_entropy(labels, logits) * [1,2,3]) / (1+2+3)
                #      = sum([10, 10, 0] * [1, 2, 3]) / 6 = 30 / 6
                metric_keys.MetricKeys.LOSS_MEAN: expected_loss / np.sum(weights_3x1x1),
            }, summary_str, tol)

    def test_multi_dim_weighted_train_create_loss(self):
        # Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 2]
        head = sequence_multi_class_head_with_crf_loss(n_classes=3, weight_column='weights')
        head.set_sequence_length([2, 1])

        logits = np.array([[[10, 0, 0], [12, 0, 0]],
                           [[0, 10, 0], [0, 15, 0]]], dtype=np.float32)
        labels = np.array([[[0], [1]], [[2], [1]]], dtype=np.int64)
        weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)

        # unreduced_loss = log_likelihood(labels, logits) = [[5.97, 5.97], [10.0, 0.]]
        expected_unreduced_loss = [[[5.97], [5.97]], [[10.0], [0.0]]]
        # weights are reshaped to [2, 2, 1] to match logits.
        expected_weights = [[[1.], [1.5]], [[2.], [0]]]
        # training_loss = 1*5.97 + 1.5*5.97 + 2*10.0 + 2.5*0 = 37.4
        expected_training_loss = 37.4
        training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
            features={'weights': weights},
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels)
        tol = 1e-2
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            self.assertAllClose(expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)
            self.assertAllClose(expected_unreduced_loss, unreduced_loss.eval(), rtol=tol, atol=tol)
            self.assertAllClose(expected_weights, actual_weights.eval())

    def test_multi_dim_weighted_train(self):
        # Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 2].
        head = sequence_multi_class_head_with_crf_loss(n_classes=3, weight_column='weights')
        head.set_sequence_length([2, 1])

        logits = np.array([[[10, 0, 0], [12, 0, 0]],
                           [[0, 10, 0], [0, 15, 0]]], dtype=np.float32)
        labels = np.array([[[0], [1]], [[2], [1]]], dtype=np.int64)
        weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)
        expected_train_result = 'my_train_op'

        def _train_op_fn(loss):
            return tf.string_join(
                [tf.constant(expected_train_result),
                 tf.as_string(loss, precision=2)])

        # unreduced_loss = log_likelihood(labels, logits) = [[5.97, 5.97], [10.0, 0.0]]
        # weighted_sum_loss = 1*5.97 + 1.5*5.97 + 2*10.0 + 2.5*0 = 37.44
        expected_loss = 37.44
        spec = head.create_estimator_spec(
            features={'weights': weights},
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels,
            train_op_fn=_train_op_fn)

        # Assert predictions, loss, train_op, and summaries.
        tol = 1e-2
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            loss, train_result = sess.run((spec.loss, spec.train_op))
            self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
            self.assertEqual(six.b('{0:s}{1:.2f}'.format(expected_train_result, expected_loss)), train_result)

    def test_multi_dim_train_weights_wrong_inner_dim(self):
        # Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 1].
        head = sequence_multi_class_head_with_crf_loss(n_classes=3, weight_column='weights')
        head.set_sequence_length([2, 1])
        logits = np.array([[[10, 0, 0], [12, 0, 0]],
                           [[0, 10, 0], [0, 15, 0]]], dtype=np.float32)
        labels = np.array([[[0], [1]], [[1], [2]]], dtype=np.int64)
        weights = np.array([[1.], [2.]], dtype=np.float32)

        def _no_op_train_fn(loss):
            del loss
            return tf.no_op()

        spec = head.create_estimator_spec(
            features={'weights': weights},
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels,
            train_op_fn=_no_op_train_fn)
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            with self.assertRaisesRegexp(
                    tf.errors.InvalidArgumentError,
                    r'\[logits_shape: \] \[2 2 3\] \[weights_shape: \] \[2 1\]'):
                spec.loss.eval()

    def test_multi_dim_train_weights_wrong_outer_dim(self):
        # Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 2, 3].
        head = sequence_multi_class_head_with_crf_loss(n_classes=3, weight_column='weights')
        head.set_sequence_length([2, 1])
        logits = np.array([[[10, 0, 0], [12, 0, 0]],
                           [[0, 10, 0], [0, 15, 0]]], dtype=np.float32)
        labels = np.array([[[0], [1]], [[1], [2]]], dtype=np.int64)
        weights = np.array([[[1., 1.1, 1.2], [1.5, 1.6, 1.7]],
                            [[2., 2.1, 2.2], [2.5, 2.6, 2.7]]])
        weights_placeholder = tf.placeholder(dtype=tf.float32)

        def _no_op_train_fn(loss):
            del loss
            return tf.no_op()

        spec = head.create_estimator_spec(
            features={'weights': weights_placeholder},
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels,
            train_op_fn=_no_op_train_fn)
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            with self.assertRaisesRegexp(
                    tf.errors.InvalidArgumentError,
                    r'\[logits_shape: \]\s\[2 2 3\]\s\[weights_shape: \]\s\[2 2 3\]'):
                spec.loss.eval({weights_placeholder: weights})

    def test_multi_dim_weighted_eval(self):
        # Logits of shape [2, 2, 2], labels [2, 2, 1], weights [2, 2].
        head = sequence_multi_class_head_with_crf_loss(n_classes=3, weight_column='weights')
        head.set_sequence_length([2, 1])
        logits = np.array([[[10, 0, 0], [12, 0, 0]],
                           [[0, 10, 0], [0, 15, 0]]], dtype=np.float32)
        labels = np.array([[[0], [1]], [[2], [1]]], dtype=np.int64)
        weights = np.array([[1., 1.5], [2., 2.5]], dtype=np.float32)
        expected_loss = 37.44
        # loss = log_likelihood(labels, logits) = [[5.97, 5.97], [10.0, 0.0]]
        # weighted_sum_loss = 1*5.97 + 1.5*5.97 + 2*10.0 + 2.5*0 = 34.9
        expected_unreduced_loss = 34.9  # a bit less then actual loss
        # Create estimator spec.
        spec = head.create_estimator_spec(
            features={'weights': weights},
            mode=tf.estimator.ModeKeys.EVAL,
            logits=logits,
            labels=labels)

        keys = metric_keys.MetricKeys
        expected_metrics = {
            keys.LOSS_MEAN: expected_unreduced_loss / (np.sum(weights) - 2.5),
            keys.ACCURACY: (1. * 1. + 1.5 * 0. + 2. * 0. + 2.5 * 0.) / (np.sum(weights) - 2.5),
        }

        # Assert predictions, loss, and metrics.
        tol = 1e-2
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
            update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
            loss, metrics = sess.run((spec.loss, update_ops))
            self.assertAllClose(expected_loss, loss, rtol=tol, atol=tol)
            # Check results of both update (in `metrics`) and value ops.
            print(metrics)
            self.assertAllClose(expected_metrics, metrics, rtol=tol, atol=tol)
            self.assertAllClose(expected_metrics, {k: value_ops[k].eval() for k in value_ops}, rtol=tol, atol=tol)

    def test_create_loss_sparse_sequence(self):
        head = sequence_multi_class_head_with_crf_loss(
            n_classes=3, weight_column='label_weights', label_vocabulary=['a', 'b', 'c'])
        head.set_sequence_length([3, 2, 1])

        logits = [
            [[10., 20., 30.], [10., 20., 30.], [10., 20., 30.]],
            [[10., 20., 30.], [10., 20., 30.], [10., 20., 30.]],
            [[10., 20., 30.], [10., 20., 30.], [10., 20., 30.]],
        ]
        labels = [
            [['c'], ['a'], ['b']],
            [['a'], ['b'], ['a']],
            [['b'], ['a'], ['c']],
        ]
        weights = [
            [[1.], [2.], [3.]],
            [[4.], [5.], [6.]],
            [[7.], [8.], [9.]],
        ]
        features = {
            'x': np.array((((42,),),), dtype=np.float32),
            'label_weights': weights
        }

        expected_weights = [
            [[1.], [2.], [3.]],
            [[4.], [5.], [0.]],
            [[7.], [0.], [0.]],
        ]
        expected_unreduced_loss = [
            [[9.8], [9.8], [9.8]],
            [[14.7], [14.7], [0.]],
            [[10.], [0.], [0.]],
        ]
        expected_training_loss = 286.6

        training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels)
        tol = 1e-2
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            self.assertAllClose(expected_weights, actual_weights.eval())
            print(unreduced_loss.eval())
            print(training_loss.eval())
            self.assertAllClose(expected_unreduced_loss, unreduced_loss.eval(), rtol=tol, atol=tol)
            self.assertAllClose(expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)


class RegressionHeadWithMeanSquaredErrorLossTest(tf.test.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_eval_create_loss(self):
        head = sequence_regression_head_with_mse_loss()
        head.set_sequence_length([2])
        logits = np.array((((45,), (41,),),), dtype=np.float32)
        labels = np.array((((43,), (44,),),), dtype=np.int32)
        features = {'x': np.array((((42,),),), dtype=np.float32)}
        # Create loss.
        training_loss = head.create_loss(
            features=features,
            mode=tf.estimator.ModeKeys.EVAL,
            logits=logits,
            labels=labels)[0]
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            # loss = [(43-45)^2, (44-41)] = [4, 9]
            self.assertAllClose(13., training_loss.eval())

    def test_eval_create_loss_loss_fn(self):
        # Tests head.create_loss for eval mode and custom loss_fn.
        loss = np.array([[[0., 1.], [2., 3.]]], dtype=np.float32)
        logits_input = np.array([[[-1., 1.], [-2., 2.]]], dtype=np.float32)
        labels_input = np.array([[[1., 0.], [2., -1.]]], dtype=np.float32)

        def _loss_fn(labels, logits):
            check_labels = tf.Assert(
                tf.reduce_all(tf.equal(labels, labels_input)),
                data=[labels])
            check_logits = tf.Assert(
                tf.reduce_all(tf.equal(logits, logits_input)),
                data=[logits])
            with tf.control_dependencies([check_labels, check_logits]):
                return tf.constant(loss)

        head = sequence_regression_head_with_mse_loss(label_dimension=2, loss_fn=_loss_fn)
        head.set_sequence_length([2])

        actual_training_loss = head.create_loss(
            features={'x': np.array((((42,),),), dtype=np.int32)},
            mode=tf.estimator.ModeKeys.EVAL,
            logits=logits_input,
            labels=labels_input)[0]
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            self.assertAllClose(np.sum(loss), actual_training_loss.eval())

    def test_eval_create_loss_loss_fn_wrong_shape(self):
        # Tests custom loss_fn that returns Tensor of unexpected shape.
        loss = np.array([[[1.], [2.]]], dtype=np.float32)

        def _loss_fn(labels, logits):
            del labels, logits  # Unused
            return tf.constant(loss)

        head = sequence_regression_head_with_mse_loss(label_dimension=2, loss_fn=_loss_fn)
        head.set_sequence_length([2])

        logits = np.array([[[-1., 1.], [-2., 2.]]], dtype=np.float32)
        labels = np.array([[[1., 0.], [2., -1.]]], dtype=np.float32)
        actual_training_loss = head.create_loss(
            features={'x': np.array((((42,),),), dtype=np.int32)},
            mode=tf.estimator.ModeKeys.EVAL,
            logits=logits,
            labels=labels)[0]
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            with self.assertRaisesRegexp(
                    tf.errors.InvalidArgumentError,
                    r'\[loss_fn must return Tensor of shape \[D0, D1, ... DN, 2\]\. \] '
                    r'\[logits_shape: \] \[1 2 2\] \[loss_shape: \] \[1 2 1\]'):
                actual_training_loss.eval()

    def test_eval(self):
        head = sequence_regression_head_with_mse_loss()
        head.set_sequence_length([2])
        self.assertEqual(1, head.logits_dimension)

        logits = np.array((((45,), (41,),),), dtype=np.float32)
        labels = np.array((((43,), (44,),),), dtype=np.int32)
        features = {'x': np.array((((42,),),), dtype=np.float32)}
        # Create estimator spec.
        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.EVAL,
            logits=logits,
            labels=labels)

        # Assert spec contains expected tensors.
        prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
        self.assertItemsEqual((prediction_key,), spec.predictions.keys())
        self.assertEqual(tf.float32, spec.predictions[prediction_key].dtype)
        self.assertEqual(tf.float32, spec.loss.dtype)
        self.assertItemsEqual((metric_keys.MetricKeys.LOSS_MEAN,
                               metric_keys.MetricKeys.PREDICTION_MEAN,
                               metric_keys.MetricKeys.LABEL_MEAN
                               ),
                              spec.eval_metric_ops.keys())
        self.assertIsNone(spec.train_op)
        self.assertIsNone(spec.export_outputs)
        _assert_no_hooks(self, spec)

        # Assert predictions, loss, and metrics.
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertIsNone(spec.scaffold.summary_op)
            loss_mean_value_op, loss_mean_update_op = spec.eval_metric_ops[
                metric_keys.MetricKeys.LOSS_MEAN]
            predictions, loss, loss_mean = sess.run((
                spec.predictions[prediction_key], spec.loss, loss_mean_update_op))
            self.assertAllClose(logits, predictions)
            # loss = (43-45)^2 + (44-41)^2 = 4+9 = 13
            self.assertAllClose(13., loss)
            # loss_mean = loss/2 = 13/2 = 6.5
            expected_loss_mean = 6.5
            # Check results of both update (in `loss_mean`) and value tf.
            self.assertAllClose(expected_loss_mean, loss_mean)
            self.assertAllClose(expected_loss_mean, loss_mean_value_op.eval())

    def test_eval_with_regularization_losses(self):
        head = sequence_regression_head_with_mse_loss(
            loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        head.set_sequence_length([2])
        self.assertEqual(1, head.logits_dimension)

        logits = np.array((((45,), (41,),),), dtype=np.float32)
        labels = np.array((((43,), (44,),),), dtype=np.int32)
        features = {'x': np.array((((42,),),), dtype=np.float32)}
        regularization_losses = [1.5, 0.5]
        expected_regularization_loss = 2.
        # unregularized_loss = ((43-45)^2 + (44-41)^2) / batch_size
        #                    = (4 + 9) / 2 = 6.5
        expected_unregularized_loss = 6.5
        expected_regularized_loss = (
                expected_unregularized_loss + expected_regularization_loss)
        # Create estimator spec.
        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.EVAL,
            logits=logits,
            labels=labels,
            regularization_losses=regularization_losses)

        keys = metric_keys.MetricKeys
        expected_metrics = {
            keys.LOSS_MEAN: expected_unregularized_loss,
            keys.LOSS_REGULARIZATION: expected_regularization_loss,
            keys.PREDICTION_MEAN: (45 + 41) / 2.0,
            keys.LABEL_MEAN: (43 + 44) / 2.0,
        }

        # Assert predictions, loss, and metrics.
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertIsNone(spec.scaffold.summary_op)
            value_ops = {k: spec.eval_metric_ops[k][0] for k in spec.eval_metric_ops}
            update_ops = {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
            prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
            predictions, loss, metrics = sess.run((
                spec.predictions[prediction_key], spec.loss, update_ops))
            self.assertAllClose(logits, predictions)
            self.assertAllClose(expected_regularized_loss, loss)
            # Check results of both update (in `metrics`) and value tf.
            self.assertAllClose(expected_metrics, metrics)
            self.assertAllClose(
                expected_metrics, {k: value_ops[k].eval() for k in value_ops})

    def test_train_create_loss(self):
        head = sequence_regression_head_with_mse_loss()
        head.set_sequence_length([2])
        logits = np.array((((45,), (41,),),), dtype=np.float32)
        labels = np.array((((43,), (44,),),), dtype=np.int32)
        features = {'x': np.array((((42,),),), dtype=np.float32)}
        # unreduced_loss = [(43-45)^2, (44-41)] = [4, 9]
        expected_unreduced_loss = [[[4.], [9.]]]
        # weights default to 1.
        expected_weights = np.array((((1.,), (1.,),),), dtype=np.float32)
        # training_loss = 1 * 4 + 1 * 9 = 13
        expected_training_loss = 13.
        # Create loss.
        training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels)
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            self.assertAllClose(expected_training_loss, training_loss.eval())
            self.assertAllClose(expected_unreduced_loss, unreduced_loss.eval())
            self.assertAllClose(expected_weights, actual_weights)

    def test_train_create_loss_loss_reduction(self):
        # Tests create_loss with loss_reduction.
        head = sequence_regression_head_with_mse_loss(
            loss_reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        head.set_sequence_length([2])
        logits = np.array((((45,), (41,),),), dtype=np.float32)
        labels = np.array((((43,), (44,),),), dtype=np.int32)
        features = {'x': np.array((((42,),),), dtype=np.float32)}
        # unreduced_loss = [(43-45)^2, (44-41)] = [4, 9]
        expected_unreduced_loss = [[[4.], [9.]]]
        # weights default to 1.
        expected_weights = np.array((((1.,), (1.,),),), dtype=np.float32)
        # training_loss = (1 * 4 + 1 * 9) / num_nonzero_weights
        expected_training_loss = 13. / 2.
        # Create loss.
        training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels)
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            self.assertAllClose(expected_training_loss, training_loss.eval())
            self.assertAllClose(expected_unreduced_loss, unreduced_loss.eval())
            self.assertAllClose(expected_weights, actual_weights)

    def test_train(self):
        head = sequence_regression_head_with_mse_loss()
        head.set_sequence_length([2])
        self.assertEqual(1, head.logits_dimension)

        # Create estimator spec.
        logits = np.array((((45,), (41,),),), dtype=np.float32)
        labels = np.array((((43,), (44,),),), dtype=np.int32)
        expected_train_result = b'my_train_op'
        features = {'x': np.array((((42,),),), dtype=np.float32)}
        # loss = (43-45)^2 + (44-41)^2 = 4 + 9 = 13
        expected_loss = 13

        def _train_op_fn(loss):
            with tf.control_dependencies((tf.assert_equal(
                    tf.to_float(expected_loss), tf.to_float(loss),
                    name='assert_loss'),)):
                return tf.constant(expected_train_result)

        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels,
            train_op_fn=_train_op_fn)

        # Assert spec contains expected tensors.
        prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
        self.assertItemsEqual((prediction_key,), spec.predictions.keys())
        self.assertEqual(tf.float32, spec.predictions[prediction_key].dtype)
        self.assertEqual(tf.float32, spec.loss.dtype)
        self.assertEqual({}, spec.eval_metric_ops)
        self.assertIsNotNone(spec.train_op)
        self.assertIsNone(spec.export_outputs)
        _assert_no_hooks(self, spec)

        # Assert predictions, loss, train_op, and summaries.
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertIsNotNone(spec.scaffold.summary_op)
            predictions, loss, train_result, summary_str = sess.run((
                spec.predictions[prediction_key], spec.loss, spec.train_op, spec.scaffold.summary_op))
            self.assertAllClose(logits, predictions)
            self.assertAllClose(expected_loss, loss)
            self.assertEqual(expected_train_result, train_result)
            _assert_simple_summaries(self, {
                metric_keys.MetricKeys.LOSS: expected_loss,
                # loss_mean = loss/2 = 13/2 = 6.5
                metric_keys.MetricKeys.LOSS_MEAN: 6.5,
            }, summary_str)

    def test_train_summaries_with_head_name(self):
        head = sequence_regression_head_with_mse_loss(name='some_regression_head')
        head.set_sequence_length([2])
        self.assertEqual(1, head.logits_dimension)

        # Create estimator spec.
        logits = np.array((((45,), (41,),),), dtype=np.float32)
        labels = np.array((((43,), (44,),),), dtype=np.int32)
        features = {'x': np.array((((42,),),), dtype=np.float32)}
        # loss = (43-45)^2 + (44-41)^2 = 4 + 9 = 13
        expected_loss = 13

        def _train_op_fn(loss):
            del loss
            return tf.no_op()

        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels,
            train_op_fn=_train_op_fn)

        # Assert summaries.
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertIsNotNone(spec.scaffold.summary_op)
            summary_str = sess.run(spec.scaffold.summary_op)
            _assert_simple_summaries(
                self,
                {
                    '{}/some_regression_head'.format(metric_keys.MetricKeys.LOSS):
                        expected_loss,
                    # loss_mean = loss/2 = 13/2 = 6.5
                    '{}/some_regression_head'
                        .format(metric_keys.MetricKeys.LOSS_MEAN):
                        6.5,
                },
                summary_str)

    def test_train_with_regularization_losses(self):
        head = sequence_regression_head_with_mse_loss(
            loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
        head.set_sequence_length([2])
        self.assertEqual(1, head.logits_dimension)

        # Create estimator spec.
        logits = np.array((((45,), (41,),),), dtype=np.float32)
        labels = np.array((((43,), (44,),),), dtype=np.int32)
        expected_train_result = b'my_train_op'
        features = {'x': np.array((((42,),),), dtype=np.float32)}
        regularization_losses = [1.5, 0.5]
        expected_regularization_loss = 2.
        # unregularized_loss = ((43-45)^2 + (44-41)^2) / batch_size
        #                    = (4 + 9) / 2 = 6.5
        # loss = unregularized_loss + regularization_loss = 8.5
        expected_loss = 8.5

        def _train_op_fn(loss):
            with tf.control_dependencies((tf.assert_equal(
                    tf.to_float(expected_loss), tf.to_float(loss),
                    name='assert_loss'),)):
                return tf.constant(expected_train_result)

        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels,
            train_op_fn=_train_op_fn,
            regularization_losses=regularization_losses)

        # Assert predictions, loss, train_op, and summaries.
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertIsNotNone(spec.scaffold.summary_op)
            prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
            predictions, loss, train_result, summary_str = sess.run((
                spec.predictions[prediction_key], spec.loss, spec.train_op, spec.scaffold.summary_op))
            self.assertAllClose(logits, predictions)
            self.assertAllClose(expected_loss, loss)
            self.assertEqual(expected_train_result, train_result)
            _assert_simple_summaries(self, {
                metric_keys.MetricKeys.LOSS: expected_loss,
                metric_keys.MetricKeys.LOSS_REGULARIZATION: (
                    expected_regularization_loss),
            }, summary_str)

    def test_weighted_multi_example_eval(self):
        # 1d label, 3 examples, 1 batch.
        head = sequence_regression_head_with_mse_loss(weight_column='label_weights')
        head.set_sequence_length([1, 1, 1])
        self.assertEqual(1, head.logits_dimension)

        # Create estimator spec.
        logits = np.array((((45,),), ((41,),), ((44,),)), dtype=np.int32)
        spec = head.create_estimator_spec(
            features={
                'x': np.array((((42,),), ((43,),), ((44,),)), dtype=np.int32),
                'label_weights': np.array((((1.,),), ((.1,),), ((1.5,),)), dtype=np.float32),
            },
            mode=tf.estimator.ModeKeys.EVAL,
            logits=logits,
            labels=np.array((((35,),), ((42,),), ((45,),)), dtype=np.int32))

        # Assert spec contains expected tensors.
        prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
        self.assertItemsEqual((prediction_key,), spec.predictions.keys())
        self.assertEqual(tf.float32, spec.predictions[prediction_key].dtype)
        self.assertEqual(tf.float32, spec.loss.dtype)
        self.assertItemsEqual((metric_keys.MetricKeys.LOSS_MEAN,
                               metric_keys.MetricKeys.PREDICTION_MEAN,
                               metric_keys.MetricKeys.LABEL_MEAN
                               ),
                              spec.eval_metric_ops.keys())
        self.assertIsNone(spec.train_op)
        self.assertIsNone(spec.export_outputs)
        _assert_no_hooks(self, spec)

        # Assert predictions, loss, and metrics.
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertIsNone(spec.scaffold.summary_op)
            loss_mean_value_op, loss_mean_update_op = spec.eval_metric_ops[
                metric_keys.MetricKeys.LOSS_MEAN]
            predictions, loss, loss_mean = sess.run((spec.predictions[prediction_key], spec.loss, loss_mean_update_op))
            self.assertAllClose(logits, predictions)
            # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
            self.assertAllClose(101.6, loss)
            # loss_mean = loss/(1+.1+1.5) = 101.6/2.6 = 39.0769231
            expected_loss_mean = 39.0769231
            # Check results of both update (in `loss_mean`) and value tf.
            self.assertAllClose(expected_loss_mean, loss_mean)
            self.assertAllClose(expected_loss_mean, loss_mean_value_op.eval())

    def test_weight_with_numeric_column(self):
        # 1d label, 3 examples, 1 batch.
        head = sequence_regression_head_with_mse_loss(
            weight_column=tf.feature_column.numeric_column('label_weights', normalizer_fn=lambda x: x + 1.))
        head.set_sequence_length([1, 1, 1])

        # Create estimator spec.
        logits = np.array((((45,),), ((41,),), ((44,),)), dtype=np.int32)
        spec = head.create_estimator_spec(
            features={
                'x': np.array((((42,),), ((43,),), ((44,),)), dtype=np.int32),
                'label_weights': np.array((((0.,),), ((-0.9,),), ((0.5,),)), dtype=np.float32),
            },
            mode=tf.estimator.ModeKeys.EVAL,
            logits=logits,
            labels=np.array((((35,),), ((42,),), ((45,),)), dtype=np.int32))

        # Assert loss.
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            loss = sess.run(spec.loss)
            # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
            self.assertAllClose(101.6, loss)

    def test_weighted_multi_example_train(self):
        # 1d label, 3 examples, 1 batch.
        head = sequence_regression_head_with_mse_loss(weight_column='label_weights')
        head.set_sequence_length([1, 1, 1])
        self.assertEqual(1, head.logits_dimension)

        # Create estimator spec.
        logits = np.array((((45,),), ((41,),), ((44,),)), dtype=np.int32)
        expected_train_result = b'my_train_op'
        # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
        expected_loss = 101.6

        def _train_op_fn(loss):
            with tf.control_dependencies((tf.assert_equal(
                    tf.to_float(expected_loss), tf.to_float(loss),
                    name='assert_loss'),)):
                return tf.constant(expected_train_result)

        spec = head.create_estimator_spec(
            features={
                'x': np.array((((42,),), ((43,),), ((44,),)), dtype=np.int32),
                'label_weights': np.array((((1.,),), ((.1,),), ((1.5,),)), dtype=np.float64),
            },
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=np.array((((35,),), ((42,),), ((45,),)), dtype=np.int32),
            train_op_fn=_train_op_fn)

        # Assert spec contains expected tensors.
        prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
        self.assertItemsEqual((prediction_key,), spec.predictions.keys())
        self.assertEqual(tf.float32, spec.predictions[prediction_key].dtype)
        self.assertEqual(tf.float32, spec.loss.dtype)
        self.assertEqual({}, spec.eval_metric_ops)
        self.assertIsNotNone(spec.train_op)
        self.assertIsNone(spec.export_outputs)
        _assert_no_hooks(self, spec)

        # Assert predictions, loss, train_op, and summaries.
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertIsNotNone(spec.scaffold.summary_op)
            predictions, loss, train_result, summary_str = sess.run((
                spec.predictions[prediction_key], spec.loss, spec.train_op,
                spec.scaffold.summary_op))
            self.assertAllClose(logits, predictions)
            self.assertAllClose(expected_loss, loss)
            self.assertEqual(expected_train_result, train_result)
            _assert_simple_summaries(self, {
                metric_keys.MetricKeys.LOSS: expected_loss,
                # loss_mean = loss/(1+.1+1.5) = 101.6/2.6 = 39.0769231
                metric_keys.MetricKeys.LOSS_MEAN: 39.0769231,
            }, summary_str)

    def test_train_one_dim_create_loss(self):
        # Tests create_loss with 1D labels and weights (shape [batch_size]).
        head = sequence_regression_head_with_mse_loss(weight_column='label_weights')
        head.set_sequence_length([3])
        logits = np.array((((45,), (41,), (44,)),), dtype=np.float32)
        x_feature_rank_1 = np.array(((42., 43., 44.,),), dtype=np.float32)
        weight_rank_1 = np.array(((1., .1, 1.5,),), dtype=np.float64)
        labels_rank_1 = np.array(((35., 42., 45.,),))
        # unreduced_loss = [(35-45)^2, (42-41)^2, (45-44)^2] = [100, 1, 1].
        expected_unreduced_loss = [[[100.], [1.], [1.]]]
        # weights are reshaped to [3, 1] to match logits.
        expected_weights = [[[1.], [.1], [1.5]]]
        # training_loss = 100 * 1 + 1 * .1 + 1.5 * 1 = 101.6
        expected_training_loss = 101.6
        features = {'x': x_feature_rank_1, 'label_weights': weight_rank_1}
        # Create loss.
        training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels_rank_1)
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            self.assertAllClose(expected_training_loss, training_loss.eval())
            self.assertAllClose(expected_unreduced_loss, unreduced_loss.eval())
            self.assertAllClose(expected_weights, actual_weights.eval())

    def test_train_one_dim(self):
        # Tests train with 1D labels and weights (shape [batch_size]).
        head = sequence_regression_head_with_mse_loss(weight_column='label_weights')
        head.set_sequence_length([3])
        self.assertEqual(1, head.logits_dimension)

        # Create estimator spec.
        logits = np.array((((45,), (41,), (44,)),), dtype=np.float32)
        expected_train_result = b'my_train_op'
        # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
        expected_loss = 101.6

        def _train_op_fn(loss):
            with tf.control_dependencies((tf.assert_equal(
                    tf.to_float(expected_loss), tf.to_float(loss),
                    name='assert_loss'),)):
                return tf.constant(expected_train_result)

        x_feature_rank_1 = np.array(((42., 43., 44.,),), dtype=np.float32)
        weight_rank_1 = np.array(((1., .1, 1.5,),), dtype=np.float64)
        labels_rank_1 = np.array(((35., 42., 45.,),))
        features = {'x': x_feature_rank_1, 'label_weights': weight_rank_1}
        self.assertEqual((1, 3,), x_feature_rank_1.shape)
        self.assertEqual((1, 3,), weight_rank_1.shape)
        self.assertEqual((1, 3,), labels_rank_1.shape)

        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels_rank_1,
            train_op_fn=_train_op_fn)

        # Assert spec contains expected tensors.
        prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
        self.assertItemsEqual((prediction_key,), spec.predictions.keys())
        self.assertEqual(tf.float32, spec.predictions[prediction_key].dtype)
        self.assertEqual(tf.float32, spec.loss.dtype)
        self.assertEqual({}, spec.eval_metric_ops)
        self.assertIsNotNone(spec.train_op)
        self.assertIsNone(spec.export_outputs)
        _assert_no_hooks(self, spec)

        # Assert predictions, loss, train_op, and summaries.
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertIsNotNone(spec.scaffold.summary_op)
            predictions, loss, train_result, summary_str = sess.run((
                spec.predictions[prediction_key], spec.loss, spec.train_op,
                spec.scaffold.summary_op))
            self.assertAllClose(logits, predictions)
            self.assertAllClose(expected_loss, loss)
            self.assertEqual(expected_train_result, train_result)
            _assert_simple_summaries(self, {
                metric_keys.MetricKeys.LOSS: expected_loss,
                # loss_mean = loss/(1+.1+1.5) = 101.6/2.6 = 39.0769231
                metric_keys.MetricKeys.LOSS_MEAN: 39.0769231,
            }, summary_str)

    def test_weighted_multi_value_eval_create_loss(self):
        # 3d label, 1 example, 1 batch.
        head = sequence_regression_head_with_mse_loss(
            weight_column='label_weights', label_dimension=3)
        head.set_sequence_length([1])
        logits = np.array((((45., 41., 44.),),))
        labels = np.array((((35., 42., 45.),),))
        features = {
            'x': np.array((((42., 43., 44.),),)),
            'label_weights': np.array((((1., .1, 1.5),),))
        }
        # Create loss.
        training_loss = head.create_loss(
            features=features,
            mode=tf.estimator.ModeKeys.EVAL,
            logits=logits,
            labels=labels)[0]
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            # loss = [(35-45)^2, (42-41)^2, (45-44)^2] = [100, 1, 1].
            # weighted sum loss = 1 * 100 + .1 * 1 + 1.5 * 1 = 101.6
            self.assertAllClose(101.6, training_loss.eval())

    def test_weighted_multi_value_eval(self):
        # 3d label, 1 example, 1 batch.
        head = sequence_regression_head_with_mse_loss(
            weight_column='label_weights', label_dimension=3)
        head.set_sequence_length([1])
        self.assertEqual(3, head.logits_dimension)

        logits = np.array((((45., 41., 44.),),))
        labels = np.array((((35., 42., 45.),),))
        features = {
            'x': np.array((((42., 43., 44.),),)),
            'label_weights': np.array((((1., .1, 1.5),),))
        }
        # Create estimator spec.
        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.EVAL,
            logits=logits,
            labels=labels)

        # Assert spec contains expected tensors.
        prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
        self.assertItemsEqual((prediction_key,), spec.predictions.keys())
        self.assertEqual(tf.float32, spec.predictions[prediction_key].dtype)
        self.assertEqual(tf.float32, spec.loss.dtype)
        self.assertItemsEqual((metric_keys.MetricKeys.LOSS_MEAN,
                               metric_keys.MetricKeys.PREDICTION_MEAN,
                               metric_keys.MetricKeys.LABEL_MEAN
                               ),
                              spec.eval_metric_ops.keys())
        self.assertIsNone(spec.train_op)
        self.assertIsNone(spec.export_outputs)
        _assert_no_hooks(self, spec)

        # Assert predictions, loss, and metrics.
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertIsNone(spec.scaffold.summary_op)
            loss_mean_value_op, loss_mean_update_op = spec.eval_metric_ops[
                metric_keys.MetricKeys.LOSS_MEAN]
            predictions, loss, loss_mean = sess.run((
                spec.predictions[prediction_key], spec.loss, loss_mean_update_op))
            self.assertAllClose(logits, predictions)
            # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
            self.assertAllClose(101.6, loss)
            # loss_mean = loss/(1+.1+1.5) = 101.6/2.6 = 39.076923
            expected_loss_mean = 39.076923
            # Check results of both update (in `loss_mean`) and value tf.
            self.assertAllClose(expected_loss_mean, loss_mean)
            self.assertAllClose(expected_loss_mean, loss_mean_value_op.eval())

    def test_weighted_multi_value_train_create_loss(self):
        # 3d label, 1 example, 1 batch.
        head = sequence_regression_head_with_mse_loss(
            weight_column='label_weights', label_dimension=3)
        head.set_sequence_length([1])
        logits = np.array((((45., 41., 44.),),))
        labels = np.array((((35., 42., 45.),),))
        features = {
            'x': np.array((((42., 43., 44.),),)),
            'label_weights': np.array((((1., .1, 1.5),),))
        }
        # Create loss.
        training_loss = head.create_loss(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels)[0]
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            # loss = [(35-45)^2, (42-41)^2, (45-44)^2] = [100, 1, 1].
            # weighted sum loss = 1 * 100 + .1 * 1 + 1.5 * 1 = 101.6
            self.assertAllClose(101.6, training_loss.eval())

    def test_weighted_multi_value_train(self):
        # 3d label, 1 example, 1 batch.
        head = sequence_regression_head_with_mse_loss(
            weight_column='label_weights', label_dimension=3)
        head.set_sequence_length([1])
        self.assertEqual(3, head.logits_dimension)

        logits = np.array((((45., 41., 44.),),))
        labels = np.array((((35., 42., 45.),),))
        expected_train_result = b'my_train_op'
        # loss = 1*(35-45)^2 + .1*(42-41)^2 + 1.5*(45-44)^2 = 100+.1+1.5 = 101.6
        expected_loss = 101.6

        def _train_op_fn(loss):
            with tf.control_dependencies((tf.assert_equal(
                    tf.to_float(expected_loss), tf.to_float(loss),
                    name='assert_loss'),)):
                return tf.constant(expected_train_result)

        features = {
            'x': np.array((((42., 43., 44.),),)),
            'label_weights': np.array((((1., .1, 1.5),),))
        }
        # Create estimator spec.
        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels,
            train_op_fn=_train_op_fn)

        # Assert spec contains expected tensors.
        prediction_key = prediction_keys.PredictionKeys.PREDICTIONS
        self.assertItemsEqual((prediction_key,), spec.predictions.keys())
        self.assertEqual(tf.float32, spec.predictions[prediction_key].dtype)
        self.assertEqual(tf.float32, spec.loss.dtype)
        self.assertEqual({}, spec.eval_metric_ops)
        self.assertIsNotNone(spec.train_op)
        self.assertIsNone(spec.export_outputs)
        _assert_no_hooks(self, spec)

        # Evaluate predictions, loss, train_op, and summaries.
        with self.test_session() as sess:
            _initialize_variables(self, spec.scaffold)
            self.assertIsNotNone(spec.scaffold.summary_op)
            predictions, loss, train_result, summary_str = sess.run((
                spec.predictions[prediction_key], spec.loss, spec.train_op,
                spec.scaffold.summary_op))
            self.assertAllClose(logits, predictions)
            self.assertAllClose(expected_loss, loss)
            self.assertEqual(expected_train_result, train_result)
            _assert_simple_summaries(self, {
                metric_keys.MetricKeys.LOSS: expected_loss,
                # loss_mean = loss/(1+.1+1.5) = 101.6/2.6 = 39.076923
                metric_keys.MetricKeys.LOSS_MEAN: 39.076923,
            }, summary_str)

    def test_weighted_multi_batch_eval(self):
        # 1d label, 1 example, 3 batches.
        head = sequence_regression_head_with_mse_loss(weight_column='label_weights')
        head.set_sequence_length([1])  # because batched
        self.assertEqual(1, head.logits_dimension)

        # Create estimator spec.
        logits = np.array((((45.,),), ((41.,),), ((44.,),)))
        input_fn = numpy_io.numpy_input_fn(
            x={
                'x': np.array((((42.,),), ((43.,),), ((44.,),))),
                'label_weights': np.array((((1.,),), ((.1,),), ((1.5,),))),
                # 'logits' is not a feature, but we use `numpy_input_fn` to make a
                # batched version of it, and pop it off before passing to
                # `create_estimator_spec`.
                'logits': logits,
            },
            y=np.array((((35.,),), ((42.,),), ((45.,),))),
            batch_size=1,
            num_epochs=1,
            shuffle=False)
        batched_features, batched_labels = input_fn()
        batched_logits = batched_features.pop('logits')
        spec = head.create_estimator_spec(
            features=batched_features,
            mode=tf.estimator.ModeKeys.EVAL,
            logits=batched_logits,
            labels=batched_labels,
            train_op_fn=None)

        # losses = [1*(35-45)^2, .1*(42-41)^2, 1.5*(45-44)^2] = [100, .1, 1.5]
        # loss = sum(losses) = 100+.1+1.5 = 101.6
        # loss_mean = loss/(1+.1+1.5) = 101.6/2.6 = 39.076923
        expected_metrics = {
            metric_keys.MetricKeys.LOSS_MEAN:
                39.076923,
            metric_keys.MetricKeys.PREDICTION_MEAN:
                (45 + 41 * 0.1 + 44 * 1.5) / 2.6,
            metric_keys.MetricKeys.LABEL_MEAN: (35 + 42 * 0.1 + 45 * 1.5) / 2.6,
        }

        # Assert spec contains expected tensors.
        self.assertEqual(tf.float32, spec.loss.dtype)
        self.assertItemsEqual(expected_metrics.keys(), spec.eval_metric_ops.keys())
        self.assertIsNone(spec.train_op)
        _assert_no_hooks(self, spec)

        with self.test_session() as sess:
            # Finalize graph and initialize variables.
            _initialize_variables(self, spec.scaffold)
            self.assertIsNotNone(spec.scaffold.summary_op)
            queue_runner_impl.start_queue_runners()

            # Run tensors for `steps` steps.
            steps = len(logits)
            results = tuple([sess.run((
                spec.loss,
                # The `[1]` gives us the metric update op.
                {k: spec.eval_metric_ops[k][1] for k in spec.eval_metric_ops}
            )) for _ in range(steps)])

            # Assert losses and metrics.
            self.assertAllClose((100, .1, 1.5), [r[0] for r in results])
            # For metrics, check results of both update (in `results`) and value tf.
            # Note: we only check the result of the last step for streaming metrics.
            self.assertAllClose(expected_metrics, results[steps - 1][1])
            self.assertAllClose(expected_metrics, {k: spec.eval_metric_ops[k][0].eval() for k in spec.eval_metric_ops})

    def test_weighted_multi_batch_train(self):
        # 1d label, 1 example, 3 batches.
        head = sequence_regression_head_with_mse_loss(weight_column='label_weights')
        head.set_sequence_length([1])  # because batched
        self.assertEqual(1, head.logits_dimension)

        # Create estimator spec.
        logits = np.array((((45.,),), ((41.,),), ((44.,),)))
        input_fn = numpy_io.numpy_input_fn(
            x={
                'x': np.array((((42.,),), ((43.,),), ((44.,),))),
                'label_weights': np.array((((1.,),), ((.1,),), ((1.5,),))),
                # 'logits' is not a feature, but we use `numpy_input_fn` to make a
                # batched version of it, and pop it off before passing to
                # `create_estimator_spec`.
                'logits': logits,
            },
            y=np.array((((35.,),), ((42.,),), ((45.,),))),
            batch_size=1,
            num_epochs=1,
            shuffle=False)
        batched_features, batched_labels = input_fn()
        batched_logits = batched_features.pop('logits')
        spec = head.create_estimator_spec(
            features=batched_features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=batched_logits,
            labels=batched_labels,
            train_op_fn=lambda loss: loss * -7.)

        # Assert spec contains expected tensors.
        self.assertEqual(tf.float32, spec.loss.dtype)
        self.assertIsNotNone(spec.train_op)

        with self.test_session() as sess:
            # Finalize graph and initialize variables.
            _initialize_variables(self, spec.scaffold)
            self.assertIsNotNone(spec.scaffold.summary_op)
            queue_runner_impl.start_queue_runners()
            results = tuple([sess.run((spec.loss, spec.train_op)) for _ in range(len(logits))])

            # losses = [1*(35-45)^2, .1*(42-41)^2, 1.5*(45-44)^2] = [100, .1, 1.5]
            expected_losses = np.array((100, .1, 1.5))
            self.assertAllClose(expected_losses, [r[0] for r in results])
            self.assertAllClose(expected_losses * -7., [r[1] for r in results])

    def test_multi_dim_weighted_train_create_loss(self):
        # Logits, labels of shape [2, 2, 3], weight shape [2, 2].
        label_dimension = 3
        head = sequence_regression_head_with_mse_loss(
            weight_column='label_weights', label_dimension=label_dimension)
        head.set_sequence_length([2, 2])
        logits = np.array([[[00., 01., 02.], [10., 11., 12.]],
                           [[20., 21., 22.], [30., 31., 32.]]])
        labels = np.array([[[01., 02., 03.], [12., 13., 14.]],
                           [[23., 24., 25.], [34., 35., 36.]]])
        weights = np.array([[1., 1.5], [2., 2.5]])
        expected_unreduced_loss = [[[1., 1., 1.], [4., 4., 4.]],
                                   [[9., 9., 9.], [16., 16., 16.]]]
        expected_training_loss = np.sum(
            np.array([[[1. * x for x in [1., 1., 1.]],
                       [1.5 * x for x in [4., 4., 4.]]],
                      [[2. * x for x in [9., 9., 9.]],
                       [2.5 * x for x in [16., 16., 16.]]]]))
        # Weights are expanded to [2, 2, 1] to match logits.
        expected_weights = [[[1.], [1.5]], [[2.], [2.5]]]
        # Create loss.
        training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
            features={'label_weights': weights},
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels)
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            self.assertAllClose(expected_training_loss, training_loss.eval())
            self.assertAllClose(expected_unreduced_loss, unreduced_loss.eval())
            self.assertAllClose(expected_weights, actual_weights.eval())

    def test_multi_dim_weighted_train(self):
        # Logits, labels of shape [2, 2, 3], weight shape [2, 2].
        head = sequence_regression_head_with_mse_loss(
            weight_column='label_weights', label_dimension=3)
        head.set_sequence_length([2, 2])
        logits = np.array([[[00., 01., 02.], [10., 11., 12.]],
                           [[20., 21., 22.], [30., 31., 32.]]])
        labels = np.array([[[01., 02., 03.], [12., 13., 14.]],
                           [[23., 24., 25.], [34., 35., 36.]]])
        expected_train_result = b'my_train_op'
        features = {
            'label_weights': np.array([[1., 1.5], [2., 2.5]]),
        }
        # loss = 1*3*1^2 + 1.5*3*2^2 + 2*3*3^2 +2.5*3*4^2 = 195
        expected_loss = 195.

        # Create estimator spec.
        def _train_op_fn(loss):
            with tf.control_dependencies((tf.assert_equal(
                    tf.to_float(expected_loss), tf.to_float(loss),
                    name='assert_loss'),)):
                return tf.constant(expected_train_result)

        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels,
            train_op_fn=_train_op_fn)
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            self.assertAllClose(expected_loss, spec.loss.eval())

        def test_multi_dim_train_weights_wrong_inner_dim(self):
            """Logits, labels of shape [2, 2, 3], weight shape [2, 1]."""
            head = sequence_regression_head_with_mse_loss(
                weight_column='label_weights', label_dimension=3)
            head.set_sequence_length([2, 2])
            logits = np.array([[[00., 01., 02.], [10., 11., 12.]],
                               [[20., 21., 22.], [30., 31., 32.]]])
            labels = np.array([[[01., 02., 03.], [12., 13., 14.]],
                               [[23., 24., 25.], [34., 35., 36.]]])
            features = {
                'label_weights': np.array([[1.], [2]]),
            }

            def _no_op_train_fn(loss):
                del loss
                return tf.no_op()

            spec = head.create_estimator_spec(
                features=features,
                mode=tf.estimator.ModeKeys.TRAIN,
                logits=logits,
                labels=labels,
                train_op_fn=_no_op_train_fn)
            with self.test_session():
                _initialize_variables(self, monitored_session.Scaffold())
                with self.assertRaisesRegexp(
                        tf.errors.InvalidArgumentError,
                        r'\[logits_shape: \] \[2 2 3\] \[weights_shape: \] \[2 1\]'):
                    spec.loss.eval()

    def test_multi_dim_train_weights_wrong_outer_dim(self):
        # Logits, labels of shape [2, 2, 3], weight shape [2, 2, 2].
        head = sequence_regression_head_with_mse_loss(
            weight_column='label_weights', label_dimension=3)
        head.set_sequence_length([2, 2])
        logits = np.array([[[00., 01., 02.], [10., 11., 12.]],
                           [[20., 21., 22.], [30., 31., 32.]]])
        labels = np.array([[[01., 02., 03.], [12., 13., 14.]],
                           [[23., 24., 25.], [34., 35., 36.]]])
        weights_placeholder = tf.placeholder(dtype=tf.float32)
        features = {
            'label_weights': weights_placeholder,
        }

        def _no_op_train_fn(loss):
            del loss
            return tf.no_op()

        spec = head.create_estimator_spec(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels,
            train_op_fn=_no_op_train_fn)
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            with self.assertRaisesRegexp(
                    tf.errors.InvalidArgumentError,
                    r'\[logits_shape: \]\s\[2 2 3\]\s\[weights_shape: \]\s\[2 2 2\]'):
                spec.loss.eval({
                    weights_placeholder: np.array([[[1., 1.1], [1.5, 1.6]],
                                                   [[2., 2.1], [2.5, 2.6]]])})

    def test_create_loss_sparse_sequence(self):
        head = sequence_regression_head_with_mse_loss(weight_column='label_weights', label_dimension=1)
        head.set_sequence_length([3, 2, 1])

        logits = [
            [[10.], [10.], [10.]],
            [[10.], [10.], [10.]],
            [[10.], [10.], [10.]],
        ]
        labels = [
            [[20.], [10.], [20.]],
            [[10.], [20.], [10.]],
            [[20.], [10.], [20.]],
        ]
        weights = [
            [[1.], [2.], [3.]],
            [[4.], [5.], [6.]],
            [[7.], [8.], [9.]],
        ]
        features = {
            'x': np.array((((42,),),), dtype=np.float32),
            'label_weights': weights
        }

        expected_weights = [
            [[1.], [2.], [3.]],
            [[4.], [5.], [0.]],
            [[7.], [0.], [0.]],
        ]
        expected_unreduced_loss = [
            [[100.], [0.], [100.]],
            [[0.], [100.], [0.]],
            [[100.], [0.], [0.]],
        ]
        expected_training_loss = 1600.

        training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels)
        tol = 1e-2
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            self.assertAllClose(expected_weights, actual_weights.eval())
            self.assertAllClose(expected_unreduced_loss, unreduced_loss.eval(), rtol=tol, atol=tol)
            self.assertAllClose(expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)


class BinaryLogisticHeadWithSigmoidCrossEntropyLossTest(tf.test.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_create_loss_sparse_sequence(self):
        # Tests create_loss with 1D labels and weights (shape [batch_size]).
        head = sequence_binary_classification_head_with_sigmoid(
            weight_column='label_weights', label_vocabulary=['a', 'b'])
        head.set_sequence_length([3, 2, 1])

        logits = [
            [[-45.], [-45.], [45.]],
            [[-45.], [45.], [-45.]],
            [[45.], [-45.], [-45.]],
        ]
        labels = [
            [['b'], ['a'], ['b']],
            [['a'], ['b'], ['a']],
            [['b'], ['a'], ['b']],
        ]
        weights = [
            [[1.], [2.], [3.]],
            [[4.], [5.], [6.]],
            [[7.], [8.], [9.]],
        ]
        features = {
            'x': np.array((((42,),),), dtype=np.float32),
            'label_weights': weights
        }

        expected_weights = [
            [[1.], [2.], [3.]],
            [[4.], [5.], [0.]],
            [[7.], [0.], [0.]],
        ]
        expected_unreduced_loss = [
            [[45.], [0.], [0.]],
            [[0.], [0.], [0.]],
            [[0.], [0.], [0.]],
        ]
        expected_training_loss = 45.

        training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels)
        tol = 1e-2
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            self.assertAllClose(expected_weights, actual_weights.eval())
            self.assertAllClose(expected_unreduced_loss, unreduced_loss.eval(), rtol=tol, atol=tol)
            self.assertAllClose(expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)


class MultiClassHeadWithSoftmaxCrossEntropyLossTest(tf.test.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_create_loss_sparse_sequence(self):
        head = sequence_multi_class_head_with_softmax(
            n_classes=3, weight_column='label_weights', label_vocabulary=['a', 'b', 'c'])
        head.set_sequence_length([3, 2, 1])

        logits = [
            [[10., 20., 30.], [10., 20., 30.], [10., 20., 30.]],
            [[10., 20., 30.], [10., 20., 30.], [10., 20., 30.]],
            [[10., 20., 30.], [10., 20., 30.], [10., 20., 30.]],
        ]
        labels = [
            [['c'], ['a'], ['b']],
            [['a'], ['b'], ['a']],
            [['b'], ['a'], ['c']],
        ]
        weights = [
            [[1.], [2.], [3.]],
            [[4.], [5.], [6.]],
            [[7.], [8.], [9.]],
        ]
        features = {
            'x': np.array((((42,),),), dtype=np.float32),
            'label_weights': weights
        }

        expected_weights = [
            [[1.], [2.], [3.]],
            [[4.], [5.], [0.]],
            [[7.], [0.], [0.]],
        ]
        expected_unreduced_loss = [
            [[0.], [20.], [10.]],
            [[20.], [10.], [0.]],
            [[10.], [0.], [0.]],
        ]
        expected_training_loss = 270.

        training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels)
        tol = 1e-2
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            self.assertAllClose(expected_weights, actual_weights.eval())
            self.assertAllClose(expected_unreduced_loss, unreduced_loss.eval(), rtol=tol, atol=tol)
            self.assertAllClose(expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)


class MultiHeadTest(tf.test.TestCase):
    def setUp(self):
        tf.reset_default_graph()

    def test_create_loss_sparse_sequence(self):
        # Tests create_loss with 1D labels and weights (shape [batch_size]).
        head1 = sequence_binary_classification_head_with_sigmoid(
            weight_column='label_weights', label_vocabulary=['a', 'b'], name='head1')
        head2 = sequence_binary_classification_head_with_sigmoid(
            weight_column='label_weights', label_vocabulary=['a', 'b'], name='head2')
        head = sequence_multi_head([head1, head2])
        head.set_sequence_length([3, 2, 1])

        logits = [
            [[-45., -45.], [-45., -45.], [45., 45.]],
            [[-45., -45.], [45., 45.], [-45., -45.]],
            [[45., 45.], [-45., -45.], [-45., -45.]],
        ]
        labels = {
            'head1': [
                [['b'], ['a'], ['b']],
                [['a'], ['b'], ['a']],
                [['b'], ['a'], ['b']],
            ],
            'head2': [
                [['b'], ['a'], ['b']],
                [['a'], ['b'], ['a']],
                [['b'], ['a'], ['b']],
            ],
        }
        weights = [
            [[1.], [2.], [3.]],
            [[4.], [5.], [6.]],
            [[7.], [8.], [9.]],
        ]
        features = {
            'x': np.array((((42,),),), dtype=np.float32),
            'label_weights': weights
        }

        expected_weights = [
            [[1.], [2.], [3.]],
            [[4.], [5.], [0.]],
            [[7.], [0.], [0.]],
        ]
        expected_unreduced_loss = [
            [[45.], [0.], [0.]],
            [[0.], [0.], [0.]],
            [[0.], [0.], [0.]],
        ]
        expected_training_loss = 45. * 2

        training_loss, unreduced_loss, actual_weights, _ = head.create_loss(
            features=features,
            mode=tf.estimator.ModeKeys.TRAIN,
            logits=logits,
            labels=labels)
        tol = 1e-2
        with self.test_session():
            _initialize_variables(self, monitored_session.Scaffold())
            self.assertAllClose(expected_weights, actual_weights['head1'].eval())
            self.assertAllClose(expected_weights, actual_weights['head2'].eval())
            self.assertAllClose(expected_unreduced_loss, unreduced_loss['head1'].eval(), rtol=tol, atol=tol)
            self.assertAllClose(expected_unreduced_loss, unreduced_loss['head2'].eval(), rtol=tol, atol=tol)
            self.assertAllClose(expected_training_loss, training_loss.eval(), rtol=tol, atol=tol)
