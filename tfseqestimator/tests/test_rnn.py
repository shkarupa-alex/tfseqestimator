from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import unittest
from ..input import build_sequence_input
from ..rnn import RnnImplementation, RnnDirection, RnnType
from ..rnn import build_dynamic_rnn, _select_last_activations
from .test_input import BuildSequenceInputTest


class BuildDynamicRnnTest(tf.test.TestCase):
    @staticmethod
    def paramsFixture():
        params = BuildSequenceInputTest.paramsFixture()
        params.add_hparam('rnn_implementation', RnnImplementation.REGULAR)
        params.add_hparam('rnn_direction', RnnDirection.UNIDIRECTIONAL)
        params.add_hparam('rnn_layers', 1)
        params.add_hparam('rnn_type', RnnType.GRU)
        params.add_hparam('rnn_units', 8)
        params.add_hparam('rnn_dropout', 0.)

        return params

    @staticmethod
    def inputsFixture():
        return build_sequence_input(
            BuildSequenceInputTest.sequenceColumnsFixture(),
            BuildSequenceInputTest.LENGTH_KEY,
            BuildSequenceInputTest.contextColumnsFixture(),
            BuildSequenceInputTest.featuresFixture(),
            BuildSequenceInputTest.paramsFixture()
        )

    def testLastOutput(self):
        # batch size = 2, padded length = 4, units number = 1
        sequence_input = tf.constant([
            [[0.0], [0.1], [0.2], [0.3]],
            [[1.0], [1.1], [1.2], [1.3]],
        ], dtype=tf.float32)
        sequence_length = tf.constant([3, 4], dtype=tf.int32)

        last_output = _select_last_activations( sequence_input, sequence_length)
        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            last_output_value = sess.run(last_output)

        self.assertAllClose(np.array([
            [0.2],
            [1.3],
        ]), last_output_value)

    def testConstructRegularForwardGRU1(self):
        sequence_input, sequence_length = self.inputsFixture()
        rnn_outputs, last_output = build_dynamic_rnn(
            sequence_input,
            sequence_length,
            self.paramsFixture(),
        )
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
        sequence_input, sequence_length = self.inputsFixture()
        rnn_outputs, last_output = build_dynamic_rnn(
            sequence_input,
            sequence_length,
            self.paramsFixture().override_from_dict({
                'rnn_direction': RnnDirection.BIDIRECTIONAL,
                'rnn_type': RnnType.LSTM
            }),
        )
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
        sequence_input, sequence_length = self.inputsFixture()
        rnn_outputs, last_output = build_dynamic_rnn(
            sequence_input,
            sequence_length,
            self.paramsFixture().override_from_dict({
                'rnn_direction': RnnDirection.STACKED,
                'rnn_type': RnnType.LSTM,
                'rnn_layers': 2,
            }),
        )
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

    @unittest.skipUnless(tf.test.is_gpu_available(cuda_only=True), 'Test only applicable when running on GPUs')
    def testConstructCudnnForwardGRU1(self):
        sequence_input, sequence_length = self.inputsFixture()
        rnn_outputs, last_output = build_dynamic_rnn(
            sequence_input,
            sequence_length,
            self.paramsFixture().override_from_dict({
                'rnn_implementation': RnnImplementation.CUDNN,
            }),
        )
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
        sequence_input, sequence_length = self.inputsFixture()
        rnn_outputs, last_output = build_dynamic_rnn(
            sequence_input,
            sequence_length,
            self.paramsFixture().override_from_dict({
                'rnn_implementation': RnnImplementation.CUDNN,
                'rnn_direction': RnnDirection.BIDIRECTIONAL,
                'rnn_type': RnnType.LSTM,
                'rnn_layers': 2,
            }),
        )
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
