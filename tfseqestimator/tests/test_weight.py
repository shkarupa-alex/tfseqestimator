from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from ..rnn import build_dynamic_rnn
from ..dense import build_logits_activations
from ..weight import make_sequence_weights
from .test_input import features_fixture
from .test_rnn import sequence_outputs, rnn_params
from .test_dense import rnn_outputs, dense_params


def dense_outputs():
    sequence_output, sequence_length = sequence_outputs()
    rnn_output = build_dynamic_rnn(
        sequence_output,
        sequence_length,
        rnn_params(),
    )
    dense_logits = build_logits_activations(
        rnn_output,
        dense_params(),
        logits_size=5
    )
    return dense_logits, sequence_length


class MaskSequenceWeightsTest(tf.test.TestCase):
    def testNoneUserWeightsActivations(self):
        features, user_weights = {}, None
        logits, length = dense_outputs()

        sequence_weights = make_sequence_weights(features, user_weights, logits, length)
        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sequence_weights_value = sess.run(sequence_weights)

        self.assertAllEqual(np.array([
            3,  # batch size
            2,  # padded length
        ]), sequence_weights_value.shape)

        self.assertAllEqual(np.array([
            [0.5, 0.5],
            [0.5, 0.5],
            [1.0, 0.0]  # Last time step masked
        ]), sequence_weights_value)

    def testProvidedUserWeightsActivations(self):
        input_features, user_weights = features_fixture(), 'sequence_weight'
        logits, length = dense_outputs()

        sequence_weights = make_sequence_weights(input_features, user_weights, logits, length)
        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sequence_weights_value = sess.run(sequence_weights)

        self.assertAllEqual(np.array([
            3,  # batch size
            2,  # padded length
        ]), sequence_weights_value.shape)

        self.assertAllEqual(np.array([
            [0.5, 1],
            [1.5, 2],
            [5.0, 0.0]  # Last time step masked
        ]), sequence_weights_value)
