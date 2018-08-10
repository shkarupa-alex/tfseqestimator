from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow import feature_column as core_columns
from tensorflow.contrib import feature_column as contrib_columns
from tensorflow.contrib.training import HParams
from ..input import build_sequence_input


def features_fixture():
    return {
        'location': tf.SparseTensor(
            indices=[[0, 0], [1, 0], [2, 0]],
            values=['west_side', 'west_side', 'nyc'],
            dense_shape=[3, 2]
        ),
        'wire_cast': tf.SparseTensor(
            indices=[
                [0, 0], [0, 1],
                [1, 0], [1, 1],
                [2, 0]],
            values=[
                b'marlo', b'stringer',
                b'omar', b'stringer',
                b'marlo'],
            dense_shape=[3, 2]
        ),
        'measurements': tf.SparseTensor(
            indices=[
                [0, 0], [0, 1], [0, 2], [0, 3],
                [1, 0], [1, 1], [1, 2], [1, 3],
                [2, 0], [2, 1]
            ],
            values=tf.random_uniform([10], seed=1),
            dense_shape=[3, 4]
        ),
    }


def sequence_columns():
    wire_cast = contrib_columns.sequence_categorical_column_with_vocabulary_list(
        'wire_cast', ['marlo', 'omar', 'stringer']
    )
    wire_cast_embedded = core_columns.embedding_column(wire_cast, dimension=8)

    measurements = contrib_columns.sequence_numeric_column('measurements', shape=(2,))

    return [measurements, wire_cast_embedded]


def context_columns():
    location = core_columns.categorical_column_with_vocabulary_list(
        'location', ['west_side', 'east_side', 'nyc']
    )
    location_onehot = core_columns.indicator_column(location)

    return [location_onehot]


class BuildSequenceInputTest(tf.test.TestCase):
    def testInputShape(self):
        partitioner = tf.min_max_variable_partitioner(max_partitions=10)
        sequence_input, _ = build_sequence_input(
            features=features_fixture(),
            sequence_columns=sequence_columns(),
            context_columns=context_columns(),
            input_partitioner=partitioner,
            sequence_dropout=0.,
            context_dropout=None,
            is_training=False,
        )
        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sequence_input_value = sess.run(sequence_input)

        self.assertAllEqual(np.array([
            3,  # expected batch size
            2,  # padded sequence length
            3 + 8 + 2  # location keys + embedding dim + measurement dimension
        ]), sequence_input_value.shape)

    def testInputDropout(self):
        sequence_input, _ = build_sequence_input(
            features=features_fixture(),
            sequence_columns=sequence_columns(),
            context_columns=context_columns(),
            input_partitioner=None,
            sequence_dropout=0.999,
            context_dropout=0.999,
            is_training=True
        )

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sequence_input_value = sess.run(sequence_input)

        self.assertAllClose(np.zeros_like(sequence_input_value), sequence_input_value)

    def testInputLength(self):
        _, sequence_length = build_sequence_input(
            features=features_fixture(),
            sequence_columns=sequence_columns(),
            context_columns=context_columns(),
            input_partitioner=None,
            sequence_dropout=0.,
            context_dropout=0.,
            is_training=False,
        )
        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sequence_length_value = sess.run(sequence_length)

        self.assertAllEqual(np.array([2, 2, 1]), sequence_length_value)
