from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from ..dense import build_logits_activations, PredictionType, DenseActivation
from ..rnn import build_dynamic_rnn
from .test_rnn import BuildDynamicRnnTest


class BuildLogitsActivationsTest(tf.test.TestCase):
    @staticmethod
    def paramsFixture():
        params = BuildDynamicRnnTest.paramsFixture()
        params.add_hparam('prediction_type', PredictionType.SINGLE)
        params.add_hparam('dense_layers', [7, 6])
        params.add_hparam('dense_activation', DenseActivation.RELU)
        params.add_hparam('dense_dropout', 0.0)

        return params

    @staticmethod
    def inputsFixture():
        sequence_input, sequence_length = BuildDynamicRnnTest.inputsFixture()
        rnn_outputs, last_output = build_dynamic_rnn(
            sequence_input,
            sequence_length,
            BuildDynamicRnnTest.paramsFixture(),
        )

        return rnn_outputs, last_output

    def testSingleClassificationActivations(self):
        rnn_outputs, last_output = self.inputsFixture()
        logits_activations = build_logits_activations(
            rnn_outputs,
            last_output,
            self.paramsFixture().override_from_dict({
                'dense_layers': [],
            }),
            logits_size=5
        )
        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            logits_activations_value = sess.run(logits_activations)

        self.assertAllEqual(np.array([
            3,  # expected batch size
            5,  # expected logits count
        ]), logits_activations_value.shape)

    def testMultipleClassificationActivations(self):
        rnn_outputs, last_output = self.inputsFixture()
        logits_activations = build_logits_activations(
            rnn_outputs,
            last_output,
            self.paramsFixture().override_from_dict({
                'prediction_type': PredictionType.MULTIPLE,
            }),
            logits_size=5
        )
        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            logits_activations_value = sess.run(logits_activations)

        self.assertAllEqual(np.array([
            3,  # expected batch size
            2,  # padded sequence length
            5,  # expected logits count
        ]), logits_activations_value.shape)

    def testSingleRegressionActivations(self):
        rnn_outputs, last_output = self.inputsFixture()
        logits_activations = build_logits_activations(
            rnn_outputs,
            last_output,
            self.paramsFixture(),
            logits_size=1
        )
        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            logits_activations_value = sess.run(logits_activations)

        self.assertAllEqual(np.array([
            3,  # expected batch size
            1,  # expected logits count
        ]), logits_activations_value.shape)

    def testMultipleRegressionActivations(self):
        rnn_outputs, last_output = self.inputsFixture()
        logits_activations = build_logits_activations(
            rnn_outputs,
            last_output,
            self.paramsFixture().override_from_dict({
                'prediction_type': PredictionType.MULTIPLE,
                'dense_layers': [],
            }),
            logits_size=1
        )
        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            logits_activations_value = sess.run(logits_activations)

        self.assertAllEqual(np.array([
            3,  # expected batch size
            2,  # padded sequence length
            1,  # expected logits count
        ]), logits_activations_value.shape)
