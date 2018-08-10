from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import shutil
import tempfile
import tensorflow as tf
import unittest
from tensorflow.contrib import feature_column as contrib_columns
from ..estimator import FullSequenceClassifier, FullSequenceRegressor
from ..estimator import SequenceItemsClassifier, SequenceItemsRegressor
from ..rnn import RnnType


def _dense_to_sparse(source, mask):
    idx = tf.where(tf.equal(mask, True))
    return tf.SparseTensor(
        idx,
        tf.gather_nd(source, idx),
        source.get_shape()
    )


def majority_dataset(num_items, max_length, batch_size):
    with tf.name_scope('dataset'):
        lengths = tf.random_uniform([num_items], minval=2, maxval=max_length, dtype=tf.int32)
        mask = tf.sequence_mask(lengths, max_length)

        inputs = tf.random_uniform([num_items, max_length], 0.0, 2.0, dtype=tf.float32)

        lookup = tf.contrib.lookup.index_to_string_table_from_tensor(['0', '1'])
        labels = tf.where(mask, inputs, tf.zeros_like(inputs, dtype=tf.float32))
        labels = tf.greater(tf.reduce_sum(labels, axis=-1), tf.to_float(lengths))
        labels = lookup.lookup(tf.to_int64(labels))

        dataset = tf.data.Dataset.from_tensor_slices((
            {'inputs': _dense_to_sparse(inputs, mask)},
            labels
        ))
        dataset = dataset.shuffle(batch_size * 2)
        dataset = dataset.batch(batch_size)

        return dataset


def mean_dataset(num_items, max_length, batch_size):
    with tf.name_scope('dataset'):
        lengths = tf.random_uniform([num_items], minval=2, maxval=max_length, dtype=tf.int32)
        mask = tf.sequence_mask(lengths, max_length)

        inputs = tf.random_uniform([num_items, max_length], minval=-1.0, maxval=1.0)

        labels = tf.where(mask, inputs, tf.zeros_like(inputs, dtype=tf.float32))
        labels = tf.reduce_sum(labels, axis=-1)
        labels = tf.divide(labels, tf.to_float(lengths))

        dataset = tf.data.Dataset.from_tensor_slices((
            {'inputs': _dense_to_sparse(inputs, mask)},
            labels
        ))
        dataset = dataset.shuffle(batch_size * 2)
        dataset = dataset.batch(batch_size)

        return dataset


def shift_dataset(num_items, max_length, batch_size):
    with tf.name_scope('dataset'):
        lengths = tf.random_uniform([num_items], minval=2, maxval=max_length, dtype=tf.int32)
        mask = tf.sequence_mask(lengths, max_length)

        random_sequence = tf.random_uniform([num_items, max_length + 1], 0, 2, dtype=tf.int32)
        inputs = tf.to_float(tf.slice(random_sequence, [0, 1], [num_items, max_length]))

        lookup = tf.contrib.lookup.index_to_string_table_from_tensor(['0', '1'])
        labels = tf.slice(random_sequence, [0, 0], [num_items, max_length])
        labels = tf.where(mask, labels, tf.zeros_like(labels, dtype=tf.int32))
        labels = lookup.lookup(tf.to_int64(labels))

        dataset = tf.data.Dataset.from_tensor_slices((
            {'inputs': _dense_to_sparse(inputs, mask)},
            labels
        ))
        dataset = dataset.shuffle(batch_size * 2)
        dataset = dataset.batch(batch_size)

        return dataset


def sine_dataset(num_items, max_length, batch_size):
    with tf.name_scope('dataset'):
        lengths = tf.random_uniform([num_items], minval=2, maxval=max_length, dtype=tf.int32)
        mask = tf.sequence_mask(lengths, max_length)

        def _sin_fn(x):
            ranger = tf.linspace(
                tf.reshape(x[0], []),
                (max_length - 1) * np.pi / 32,
                max_length + 1
            )
            return tf.sin(ranger)

        starts = tf.random_uniform([num_items], maxval=(2 * np.pi))
        sin_curves = tf.map_fn(_sin_fn, (starts,), dtype=tf.float32)

        inputs = tf.slice(sin_curves, [0, 0], [num_items, max_length])

        labels = tf.slice(sin_curves, [0, 1], [num_items, max_length])
        labels = tf.where(mask, labels, tf.zeros_like(labels, dtype=tf.float32))

        dataset = tf.data.Dataset.from_tensor_slices((
            {'inputs': _dense_to_sparse(inputs, mask)},
            labels
        ))
        dataset = dataset.shuffle(batch_size * 2)
        dataset = dataset.batch(batch_size)

        return dataset


class FullSequenceClassifierTest(tf.test.TestCase):
    def setUp(self):
        self.model_dir = tempfile.mkdtemp()
        self.export_dir = tempfile.mkdtemp()
        tf.set_random_seed(1234)
        np.random.seed(1234)

    def tearDown(self):
        shutil.rmtree(self.model_dir, ignore_errors=True)
        shutil.rmtree(self.export_dir, ignore_errors=True)

    def testLearnMajority(self):
        estimator = FullSequenceClassifier(
            label_vocabulary=['0', '1'],
            sequence_columns=[contrib_columns.sequence_numeric_column('inputs')],
            rnn_type=RnnType.REGULAR_FORWARD_LSTM,
            rnn_layers=[4],
            learning_rate=0.1,
            model_dir=self.model_dir
        )
        estimator.train(input_fn=lambda: majority_dataset(10000, 7, 16), steps=500)

        tf.set_random_seed(4321)
        accuracy = estimator.evaluate(input_fn=lambda: majority_dataset(400, 7, 16), steps=20)['accuracy']
        self.assertGreater(accuracy, 0.6, 'Accuracy should be greater than {}; got {}'.format(0.6, accuracy))

    def testExportRegularForwardNoDnn(self):
        estimator = FullSequenceClassifier(
            label_vocabulary=['0', '1'],
            sequence_columns=[contrib_columns.sequence_numeric_column('inputs')],
            rnn_type=RnnType.REGULAR_FORWARD_LSTM,
            rnn_layers=[4],
            learning_rate=0.1,
            model_dir=self.model_dir
        )
        estimator.train(input_fn=lambda: majority_dataset(10000, 7, 16), steps=500)

        feature_spec = tf.estimator.classifier_parse_example_spec(
            [contrib_columns.sequence_numeric_column('inputs')],
            label_key='label',
            label_dtype=tf.int64
        )
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        estimator.export_savedmodel(self.export_dir, serving_input_receiver_fn)
        self.assertTrue(tf.gfile.Exists(self.export_dir))


class FullSequenceRegressorTest(tf.test.TestCase):
    def setUp(self):
        self.model_dir = tempfile.mkdtemp()
        self.export_dir = tempfile.mkdtemp()
        tf.set_random_seed(1234)
        np.random.seed(1234)

    def tearDown(self):
        # shutil.rmtree(self.model_dir, ignore_errors=True)
        shutil.rmtree(self.export_dir, ignore_errors=True)

    def testLearnMean(self):
        estimator = FullSequenceRegressor(
            label_dimension=1,
            sequence_columns=[contrib_columns.sequence_numeric_column('inputs')],
            rnn_type=RnnType.REGULAR_FORWARD_GRU,
            rnn_layers=[8],
            learning_rate=0.1,
            model_dir=self.model_dir
        )
        estimator.train(input_fn=lambda: mean_dataset(10000, 50, 16), steps=500)

        tf.set_random_seed(4321)
        loss = estimator.evaluate(input_fn=lambda: mean_dataset(400, 50, 16), steps=20)['loss']
        self.assertLess(loss, 0.1, 'Loss should be less than {}; got {}'.format(0.1, loss))

    def testExportRegularBiDnn(self):
        estimator = FullSequenceRegressor(
            label_dimension=1,
            sequence_columns=[contrib_columns.sequence_numeric_column('inputs')],
            rnn_type=RnnType.REGULAR_BIDIRECTIONAL_LSTM,
            rnn_layers=[8],
            dense_layers=[-2, -3, 3, 2],
            dense_norm=True,
            model_dir=self.model_dir
        )
        estimator.train(input_fn=lambda: mean_dataset(10000, 50, 16), steps=500)

        feature_spec = tf.estimator.classifier_parse_example_spec(
            [contrib_columns.sequence_numeric_column('inputs')],
            label_key='label',
            label_dtype=tf.float32
        )
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        estimator.export_savedmodel(self.export_dir, serving_input_receiver_fn)
        self.assertTrue(tf.gfile.Exists(self.export_dir))


class SequenceItemsClassifierTest(tf.test.TestCase):
    def setUp(self):
        self.model_dir = tempfile.mkdtemp()
        self.export_dir = tempfile.mkdtemp()
        tf.set_random_seed(1234)
        np.random.seed(1234)

    def tearDown(self):
        shutil.rmtree(self.model_dir, ignore_errors=True)
        shutil.rmtree(self.export_dir, ignore_errors=True)

    def testLearnShiftByOne(self):
        estimator = SequenceItemsClassifier(
            label_vocabulary=['0', '1'],
            sequence_columns=[contrib_columns.sequence_numeric_column('inputs')],
            rnn_layers=[4],
            learning_rate=0.3,
            model_dir=self.model_dir
        )
        estimator.train(input_fn=lambda: shift_dataset(2000, 32, 16), steps=200)

        tf.set_random_seed(4321)
        accuracy = estimator.evaluate(input_fn=lambda: shift_dataset(200, 32, 16), steps=20)['accuracy']
        self.assertGreater(accuracy, 0.9, 'Accuracy should be greater than {}; got {}'.format(0.9, accuracy))

    def testExportRegularStackedNoDnn(self):
        estimator = SequenceItemsClassifier(
            label_vocabulary=['0', '1'],
            sequence_columns=[contrib_columns.sequence_numeric_column('inputs')],
            rnn_type=RnnType.REGULAR_STACKED_GRU,
            rnn_layers=[4, 3],
            learning_rate=0.3,
            model_dir=self.model_dir
        )
        estimator.train(input_fn=lambda: shift_dataset(2000, 32, 16), steps=200)

        feature_spec = tf.estimator.classifier_parse_example_spec(
            [contrib_columns.sequence_numeric_column('inputs')],
            label_key='label',
            label_dtype=tf.int64
        )
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        estimator.export_savedmodel(self.export_dir, serving_input_receiver_fn)
        self.assertTrue(tf.gfile.Exists(self.export_dir))


class SequenceItemsRegressorTest(tf.test.TestCase):
    def setUp(self):
        self.model_dir = tempfile.mkdtemp()
        self.export_dir = tempfile.mkdtemp()
        tf.set_random_seed(1234)
        np.random.seed(1234)

    def tearDown(self):
        shutil.rmtree(self.model_dir, ignore_errors=True)
        shutil.rmtree(self.export_dir, ignore_errors=True)

    def testLearnSineFunction(self):
        estimator = SequenceItemsRegressor(
            label_dimension=1,
            sequence_columns=[contrib_columns.sequence_numeric_column('inputs')],
            rnn_layers=[4],
            learning_rate=0.01,
            model_dir=self.model_dir
        )
        estimator.train(input_fn=lambda: sine_dataset(20000, 64, 16), steps=1000)

        tf.set_random_seed(4321)
        loss = estimator.evaluate(input_fn=lambda: sine_dataset(200, 64, 16), steps=20)['loss']
        self.assertLess(loss, 0.3, 'Loss should be less than {}; got {}'.format(0.3, loss))

    @unittest.skipUnless(tf.test.is_gpu_available(cuda_only=True), 'Test only applicable when running on GPUs')
    def testExportCudnnBiDnn(self):
        estimator = SequenceItemsRegressor(
            label_dimension=1,
            sequence_columns=[contrib_columns.sequence_numeric_column('inputs')],
            rnn_type=RnnType.CUDNN_BIDIRECTIONAL_GRU,
            rnn_layers=[4, 3],
            dense_layers=[3, 2],
            learning_rate=0.1,
            model_dir=self.model_dir
        )
        estimator.train(input_fn=lambda: sine_dataset(2000, 64, 8), steps=200)

        feature_spec = tf.estimator.classifier_parse_example_spec(
            [contrib_columns.sequence_numeric_column('inputs')],
            label_key='label',
            label_dtype=tf.float32
        )
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        estimator.export_savedmodel(self.export_dir, serving_input_receiver_fn)
        self.assertTrue(tf.gfile.Exists(self.export_dir))
