from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import unittest
from ..input import build_sequence_input
from ..rnn import RnnType, build_dynamic_rnn, select_last_activations
from .test_input import features_fixture, sequence_columns, context_columns


def sequence_outputs():
    return build_sequence_input(
        features=features_fixture(),
        sequence_columns=sequence_columns(),
        context_columns=context_columns(),
        input_partitioner=None,
        sequence_dropout=0.,
        context_dropout=0.,
        is_training=False,
    )


class BuildDynamicRnnTest(tf.test.TestCase):
    def testLastOutput(self):
        # batch size = 2, padded length = 4, units number = 1
        sequence_output = tf.constant([
            [[0.0], [0.1], [0.2], [0.3]],
            [[1.0], [1.1], [1.2], [1.3]],
        ], dtype=tf.float32)
        sequence_length = tf.constant([3, 4], dtype=tf.int64)
        last_output = select_last_activations(sequence_output, sequence_length)

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            last_output_value = sess.run(last_output)

        self.assertAllClose(np.array([
            [0.2],
            [1.3],
        ]), last_output_value)

    def testConstructRegularForwardGRU1(self):
        sequence_input, sequence_length = sequence_outputs()
        rnn_outputs = build_dynamic_rnn(
            sequence_input=sequence_input,
            sequence_length=sequence_length,
            rnn_type=RnnType.REGULAR_FORWARD_GRU,
            rnn_layers=[8],
            rnn_dropout=0.,
            is_training=False,
        )
        last_output = select_last_activations(rnn_outputs, sequence_length)

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            rnn_outputs_value, last_output_value = sess.run([rnn_outputs, last_output])

        self.assertAllEqual(np.array([
            3,  # expected batch size
            2,  # padded sequence length
            8  # cell size
        ]), rnn_outputs_value.shape)
        self.assertAllEqual(np.array([
            3,  # expected batch size
            8  # cell size
        ]), last_output_value.shape)

    def testConstructRegularBidirectionalLSTM1(self):
        sequence_input, sequence_length = sequence_outputs()
        rnn_outputs = build_dynamic_rnn(
            sequence_input=sequence_input,
            sequence_length=sequence_length,
            rnn_type=RnnType.REGULAR_BIDIRECTIONAL_LSTM,
            rnn_layers=[8],
            rnn_dropout=0.,
            is_training=False,
        )
        last_output = select_last_activations(rnn_outputs, sequence_length)

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            rnn_outputs_value, last_output_value = sess.run([rnn_outputs, last_output])

        self.assertAllEqual(np.array([
            3,  # expected batch size
            2,  # padded sequence length
            8 * 2  # cell size * directions
        ]), rnn_outputs_value.shape)
        self.assertAllEqual(np.array([
            3,  # expected batch size
            8 * 2  # cell size * directions
        ]), last_output_value.shape)

    def testConstructRegularStackedLSTM2(self):
        sequence_input, sequence_length = sequence_outputs()
        rnn_outputs = build_dynamic_rnn(
            sequence_input=sequence_input,
            sequence_length=sequence_length,
            rnn_type=RnnType.REGULAR_STACKED_LSTM,
            rnn_layers=[8, 4],
            rnn_dropout=0.,
            is_training=False,
        )
        last_output = select_last_activations(rnn_outputs, sequence_length)

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            rnn_outputs_value, last_output_value = sess.run([rnn_outputs, last_output])

        self.assertAllEqual(np.array([
            3,  # expected batch size
            2,  # padded sequence length
            4 * 2  # cell size * directions
        ]), rnn_outputs_value.shape)
        self.assertAllEqual(np.array([
            3,  # expected batch size
            4 * 2  # cell size * directions
        ]), last_output_value.shape)

    @unittest.skipUnless(tf.test.is_gpu_available(cuda_only=True), 'Test only applicable when running on GPUs')
    def testConstructCudnnForwardGRU1(self):
        sequence_input, sequence_length = sequence_outputs()
        rnn_outputs = build_dynamic_rnn(
            sequence_input=sequence_input,
            sequence_length=sequence_length,
            rnn_type=RnnType.CUDNN_FORWARD_GRU,
            rnn_layers=[8],
            rnn_dropout=0.,
            is_training=False,
        )
        last_output = select_last_activations(rnn_outputs, sequence_length)

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            rnn_outputs_value, last_output_value = sess.run([rnn_outputs, last_output])

        self.assertAllEqual(np.array([
            3,  # expected batch size
            2,  # padded sequence length
            8  # cell size
        ]), rnn_outputs_value.shape)
        self.assertAllEqual(np.array([
            3,  # expected batch size
            8  # cell size
        ]), last_output_value.shape)

    @unittest.skipUnless(tf.test.is_gpu_available(cuda_only=True), 'Test only applicable when running on GPUs')
    def testConstructCudnnBidirectionalLSTM2(self):
        sequence_input, sequence_length = sequence_outputs()
        rnn_outputs = build_dynamic_rnn(
            sequence_input=sequence_input,
            sequence_length=sequence_length,
            rnn_type=RnnType.CUDNN_BIDIRECTIONAL_LSTM,
            rnn_layers=[8, 8],
            rnn_dropout=0.,
            is_training=False,
        )
        last_output = select_last_activations(rnn_outputs, sequence_length)

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            rnn_outputs_value, last_output_value = sess.run([rnn_outputs, last_output])

        self.assertAllEqual(np.array([
            3,  # expected batch size
            2,  # padded sequence length
            8 * 2  # cell size * directions
        ]), rnn_outputs_value.shape)
        self.assertAllEqual(np.array([
            3,  # expected batch size
            8 * 2  # cell size * directions
        ]), last_output_value.shape)
