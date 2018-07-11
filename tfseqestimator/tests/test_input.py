from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.training import HParams
from ..input import build_sequence_input


class BuildSequenceInputTest(tf.test.TestCase):
    LENGTH_KEY = 'sequence_length'

    @staticmethod
    def featuresFixture():
        return {
            'location':
                tf.SparseTensor(
                    indices=[[0, 0], [1, 0], [2, 0]],
                    values=['west_side', 'west_side', 'nyc'],
                    dense_shape=[3, 1]
                ),
            'wire_cast':
                tf.SparseTensor(
                    indices=[[0, 0, 0], [0, 1, 0],
                             [1, 0, 0], [1, 1, 0], [1, 1, 1],
                             [2, 0, 0]],
                    values=[b'marlo', b'stringer', b'omar', b'stringer', b'marlo', b'marlo'],
                    dense_shape=[3, 2, 2]
                ),
            'measurements':
                tf.random_uniform([3, 2, 2], seed=1),
            BuildSequenceInputTest.LENGTH_KEY:
                tf.constant([2, 2, 1]),
            'sequence_weight':
                tf.constant([
                    [1, 2],
                    [3, 4],
                    [5, 6],
                ]),
        }

    @staticmethod
    def sequenceColumnsFixture():
        wire_cast = feature_column.sparse_column_with_keys('wire_cast', ['marlo', 'omar', 'stringer'])
        wire_cast_embedded = feature_column.embedding_column(wire_cast, dimension=8)

        measurements = feature_column.real_valued_column('measurements', dimension=2)

        return [measurements, wire_cast_embedded]

    @staticmethod
    def contextColumnsFixture():
        location = feature_column.sparse_column_with_keys('location', keys=['west_side', 'east_side', 'nyc'])
        location_onehot = feature_column.one_hot_column(location)

        return [location_onehot]

    @staticmethod
    def paramsFixture():
        return HParams(
            sequence_dropout=0.0,
            context_dropout=None,
        )

    def testInputShape(self):
        sequence_input, _ = build_sequence_input(
            self.sequenceColumnsFixture(),
            self.LENGTH_KEY,
            self.contextColumnsFixture(),
            self.featuresFixture(),
            self.paramsFixture()
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
        params = self.paramsFixture().override_from_dict({
            'sequence_dropout': 0.999,
            'context_dropout': 0.999,
        })
        sequence_input, _ = build_sequence_input(
            self.sequenceColumnsFixture(),
            self.LENGTH_KEY,
            self.contextColumnsFixture(),
            self.featuresFixture(),
            params,
            is_training=True
        )

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sequence_input_value = sess.run(sequence_input)

        self.assertAllClose(np.zeros([3, 2, 13]), sequence_input_value)

    def testInputLengthKey(self):
        _, sequence_length = build_sequence_input(
            self.sequenceColumnsFixture(),
            self.LENGTH_KEY,
            self.contextColumnsFixture(),
            self.featuresFixture(),
            self.paramsFixture()
        )
        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sequence_length_value = sess.run(sequence_length)

        self.assertAllEqual(np.array([2, 2, 1]), sequence_length_value)

    def testInputLengthColumn(self):
        _, sequence_length = build_sequence_input(
            self.sequenceColumnsFixture(),
            tf.feature_column.numeric_column(self.LENGTH_KEY),
            self.contextColumnsFixture(),
            self.featuresFixture(),
            self.paramsFixture()
        )
        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sequence_length_value = sess.run(sequence_length)

        self.assertAllEqual(np.array([2, 2, 1]), sequence_length_value)
