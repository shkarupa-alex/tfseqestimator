from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..dense import build_logits_activations, PredictionType
from ..weight import make_sequence_weights, _FINAL_WEIGHTS_KEY
from .test_input import BuildSequenceInputTest
from .test_rnn import BuildDynamicRnnTest
from .test_dense import BuildLogitsActivationsTest
import numpy as np
import tensorflow as tf


class MaskSequenceWeightsTest(tf.test.TestCase):
    @staticmethod
    def featuresFixture():
        return BuildSequenceInputTest.featuresFixture()

    @staticmethod
    def inputsFixture():
        _, sequence_length = BuildDynamicRnnTest.inputsFixture()
        rnn_outputs, last_output = BuildDynamicRnnTest.inputsFixture()
        logits_activations = build_logits_activations(
            rnn_outputs,
            last_output,
            BuildLogitsActivationsTest.paramsFixture().override_from_dict({
                'prediction_type': PredictionType.MULTIPLE,
            }),
            logits_size=5
        )

        return logits_activations, sequence_length

    def testNoneUserWeightsActivations(self):
        input_features, user_weights = {}, None
        logits, length = self.inputsFixture()

        features = make_sequence_weights(input_features, user_weights, logits, length)
        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            weights_value = sess.run(features[_FINAL_WEIGHTS_KEY])

        self.assertAllEqual(np.array([
            3,  # batch size
            2,  # padded length
        ]), weights_value.shape)

        self.assertAllEqual(np.array([
            [0.5, 0.5],
            [0.5, 0.5],
            [1.0, 0.0]  # Last time step masked
        ]), weights_value)

    def testProvidedUserWeightsActivations(self):
        input_features, user_weights = self.featuresFixture(), 'sequence_weight'
        logits, length = self.inputsFixture()

        features = make_sequence_weights(input_features, user_weights, logits, length)
        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            weights_value = sess.run(features[_FINAL_WEIGHTS_KEY])

        self.assertAllEqual(np.array([
            3,  # batch size
            2,  # padded length
        ]), weights_value.shape)

        self.assertAllEqual(np.array([
            [0.5, 1],
            [1.5, 2],
            [5.0, 0.0]  # Last time step masked
        ]), weights_value)
