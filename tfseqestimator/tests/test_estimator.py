from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..estimator import FullSequenceClassifier, FullSequenceRegressor
from ..estimator import SequenceItemsClassifier, SequenceItemsRegressor
from ..rnn import RnnImplementation, RnnDirection, RnnType
from tensorflow.contrib.layers.python.layers import feature_column
import numpy as np
import shutil
import tempfile
import tensorflow as tf
import unittest


class FullSequenceClassifierTest(tf.test.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @staticmethod
    def inputFixture(num_items, max_length, batch_size):
        lengths = tf.random_uniform([num_items], minval=2, maxval=max_length, dtype=tf.int32)
        mask = tf.sequence_mask(lengths, max_length)

        inputs = tf.random_uniform([num_items, max_length], 0.0, 2.0, dtype=tf.float32)

        lookup = tf.contrib.lookup.index_to_string_table_from_tensor(['0', '1'])
        labels = tf.where(mask, inputs, tf.zeros_like(inputs, dtype=tf.float32))
        labels = tf.greater(tf.reduce_sum(labels, axis=-1), tf.to_float(lengths))
        labels = lookup.lookup(tf.to_int64(labels))

        dataset = tf.data.Dataset.from_tensor_slices(({'inputs': inputs, 'lengths': lengths}, labels))
        dataset = dataset.shuffle(batch_size * 2)
        dataset = dataset.batch(batch_size)

        return dataset

    def testLearnMajority(self):
        tf.set_random_seed(1234)
        estimator = FullSequenceClassifier(
            label_vocabulary=['0', '1'],
            model_params={
                'rnn_direction': RnnDirection.UNIDIRECTIONAL,
                'rnn_units': 4,
                'learning_rate': 0.1
            },
            sequence_columns=[feature_column.real_valued_column('inputs')],
            length_column=tf.feature_column.numeric_column('lengths'),
        )
        estimator.train(input_fn=lambda: self.inputFixture(10000, 7, 16), steps=500)

        tf.set_random_seed(4321)
        accuracy = estimator.evaluate(input_fn=lambda: self.inputFixture(400, 7, 16), steps=20)['accuracy']
        self.assertGreater(accuracy, 0.6, 'Accuracy should be greater than {}; got {}'.format(0.6, accuracy))

    def testExportRegularForwardNoDnn(self):
        estimator = FullSequenceClassifier(
            label_vocabulary=['0', '1'],
            model_params={
                'rnn_units': 4,
                'learning_rate': 0.3
            },
            sequence_columns=[feature_column.real_valued_column('inputs')],
            length_column='lengths',
        )
        estimator.train(input_fn=lambda: self.inputFixture(10000, 7, 16), steps=500)

        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'inputs': tf.placeholder(dtype=tf.float32, shape=[None, None]),
            'lengths': tf.placeholder(dtype=tf.int32, shape=[None]),
        })
        estimator.export_savedmodel(self.temp_dir, serving_input_receiver_fn)


class FullSequenceRegressorTest(tf.test.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @staticmethod
    def inputFixture(num_items, max_length, batch_size):
        lengths = tf.random_uniform([num_items], minval=2, maxval=max_length, dtype=tf.int32)
        mask = tf.sequence_mask(lengths, max_length)

        inputs = tf.random_uniform([num_items, max_length], minval=-1.0, maxval=1.0)

        labels = tf.where(mask, inputs, tf.zeros_like(inputs, dtype=tf.float32))
        labels = tf.reduce_sum(labels, axis=-1)
        labels = tf.divide(labels, tf.to_float(lengths))

        dataset = tf.data.Dataset.from_tensor_slices(({'inputs': inputs, 'lengths': lengths}, labels))
        dataset = dataset.shuffle(batch_size * 2)
        dataset = dataset.batch(batch_size)

        return dataset

    def testLearnMean(self):
        tf.set_random_seed(1234)
        estimator = FullSequenceRegressor(
            label_dimension=1,
            model_params={
                'rnn_direction': RnnDirection.UNIDIRECTIONAL,
                'rnn_type': RnnType.GRU,
                'rnn_units': 8,
                'learning_rate': 0.1
            },
            sequence_columns=[feature_column.real_valued_column('inputs')],
            length_column='lengths',
        )
        estimator.train(input_fn=lambda: self.inputFixture(10000, 50, 16), steps=500)

        tf.set_random_seed(4321)
        loss = estimator.evaluate(input_fn=lambda: self.inputFixture(400, 50, 16), steps=20)['loss']
        self.assertLess(loss, 0.1, 'Loss should be less than {}; got {}'.format(0.1, loss))

    def testExportRegularBiDnn(self):
        estimator = FullSequenceRegressor(
            label_dimension=1,
            model_params={
                'rnn_direction': RnnDirection.BIDIRECTIONAL,
                'rnn_units': 8,
                'rnn_type': RnnType.GRU,
                'dense_layers': [3, 2],
                'learning_rate': 0.1
            },
            sequence_columns=[feature_column.real_valued_column('inputs')],
            length_column='lengths',
        )
        estimator.train(input_fn=lambda: self.inputFixture(10000, 50, 16), steps=500)

        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'inputs': tf.placeholder(dtype=tf.float32, shape=[None, None]),
            'lengths': tf.placeholder(dtype=tf.int32, shape=[None]),
        })
        estimator.export_savedmodel(self.temp_dir, serving_input_receiver_fn)


class SequenceItemsClassifierTest(tf.test.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @staticmethod
    def inputFixture(num_items, max_length, batch_size):
        lengths = tf.random_uniform([num_items], minval=2, maxval=max_length, dtype=tf.int32)
        mask = tf.sequence_mask(lengths, max_length)

        random_sequence = tf.random_uniform([num_items, max_length + 1], 0, 2, dtype=tf.int32)
        inputs = tf.expand_dims(tf.to_float(tf.slice(random_sequence, [0, 1], [num_items, max_length])), 2)

        lookup = tf.contrib.lookup.index_to_string_table_from_tensor(['0', '1'])
        labels = tf.slice(random_sequence, [0, 0], [num_items, max_length])
        labels = tf.where(mask, labels, tf.zeros_like(labels, dtype=tf.int32))
        labels = lookup.lookup(tf.to_int64(labels))

        dataset = tf.data.Dataset.from_tensor_slices(({'inputs': inputs, 'lengths': lengths}, labels))
        dataset = dataset.shuffle(batch_size * 2)
        dataset = dataset.batch(batch_size)

        return dataset

    def testLearnShiftByOne(self):
        tf.set_random_seed(1234)
        estimator = SequenceItemsClassifier(
            label_vocabulary=['0', '1'],
            model_params={
                'rnn_direction': RnnDirection.UNIDIRECTIONAL,
                'rnn_units': 4,
                'learning_rate': 0.3
            },
            sequence_columns=[feature_column.real_valued_column('inputs')],
            length_column='lengths',
        )
        estimator.train(input_fn=lambda: self.inputFixture(2000, 32, 16), steps=200)

        tf.set_random_seed(4321)
        accuracy = estimator.evaluate(input_fn=lambda: self.inputFixture(200, 32, 16), steps=20)['accuracy']
        self.assertGreater(accuracy, 0.9, 'Accuracy should be greater than {}; got {}'.format(0.9, accuracy))

    def testExportRegularStackedNoDnn(self):
        estimator = SequenceItemsClassifier(
            label_vocabulary=['0', '1'],
            model_params={
                'rnn_direction': RnnDirection.STACKED,
                'rnn_layers': 2,
                'rnn_units': 4,
                'learning_rate': 0.3
            },
            sequence_columns=[feature_column.real_valued_column('inputs')],
            length_column='lengths',
        )
        estimator.train(input_fn=lambda: self.inputFixture(2000, 32, 16), steps=200)

        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'inputs': tf.placeholder(dtype=tf.float32, shape=[None, None]),
            'lengths': tf.placeholder(dtype=tf.int32, shape=[None]),
        })
        estimator.export_savedmodel(self.temp_dir, serving_input_receiver_fn)


class SequenceItemsRegressorTest(tf.test.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @staticmethod
    def inputFixture(num_items, max_length, batch_size):
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

        inputs = tf.expand_dims(tf.slice(sin_curves, [0, 0], [num_items, max_length]), 2)

        labels = tf.slice(sin_curves, [0, 1], [num_items, max_length])
        labels = tf.where(mask, labels, tf.zeros_like(labels, dtype=tf.float32))

        dataset = tf.data.Dataset.from_tensor_slices(({'inputs': inputs, 'lengths': lengths}, labels))
        dataset = dataset.shuffle(batch_size * 2)
        dataset = dataset.batch(batch_size)

        return dataset

    def testLearnSineFunction(self):
        tf.set_random_seed(1234)
        estimator = SequenceItemsRegressor(
            label_dimension=1,
            model_params={
                'rnn_direction': RnnDirection.UNIDIRECTIONAL,
                'rnn_units': 4,
                'learning_rate': 0.1
            },
            sequence_columns=[feature_column.real_valued_column('inputs')],
            length_column=tf.feature_column.numeric_column('lengths'),
        )
        estimator.train(input_fn=lambda: self.inputFixture(2000, 64, 8), steps=200)

        tf.set_random_seed(4321)
        loss = estimator.evaluate(input_fn=lambda: self.inputFixture(200, 64, 8), steps=20)['loss']
        self.assertLess(loss, 0.02, 'Loss should be less than {}; got {}'.format(0.02, loss))

    @unittest.skipUnless(tf.test.is_gpu_available(cuda_only=True), 'Test only applicable when running on GPUs')
    def testExportCudnnBiDnn(self):
        estimator = SequenceItemsRegressor(
            label_dimension=1,
            model_params={
                'rnn_implementation': RnnImplementation.CUDNN,
                'rnn_direction': RnnDirection.BIDIRECTIONAL,
                'rnn_units': 4,
                'dense_layers': [3, 2],
                'learning_rate': 0.1
            },
            sequence_columns=[feature_column.real_valued_column('inputs')],
            length_column='lengths',
        )
        estimator.train(input_fn=lambda: self.inputFixture(2000, 64, 8), steps=200)

        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
            'inputs': tf.placeholder(dtype=tf.float32, shape=[None, None]),
            'lengths': tf.placeholder(dtype=tf.int32, shape=[None]),
        })
        estimator.export_savedmodel(self.temp_dir, serving_input_receiver_fn)
