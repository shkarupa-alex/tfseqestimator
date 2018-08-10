from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from ..rnn import RnnType, build_dynamic_rnn
from ..dense import build_logits_activations, apply_time_distributed
from ..weight import make_sequence_weights
from .test_rnn import sequence_outputs


def dense_outputs():
    sequence_output, sequence_length = sequence_outputs()
    rnn_output = build_dynamic_rnn(
        sequence_input=sequence_output,
        sequence_length=sequence_length,
        rnn_type=RnnType.REGULAR_FORWARD_GRU,
        rnn_layers=[8],
        rnn_dropout=0.,
        is_training=False,
    )
    dense_logits = apply_time_distributed(
        layer_producer=build_logits_activations,
        sequence_input=rnn_output,
        logits_size=5,
        dense_layers=[7],
        dense_activation='sigmoid',
        dense_dropout=0.,
        dense_norm=False,
        is_training=False,
    )

    return dense_logits, sequence_length


class MaskSequenceWeightsTest(tf.test.TestCase):
    def testNoneUserWeightsActivations(self):
        logits, length = dense_outputs()

        sequence_weights = make_sequence_weights(1., logits, length)
        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sequence_weights_value = sess.run(sequence_weights)
            print(sequence_weights_value)

        self.assertAllEqual(np.array([
            3,  # batch size
            2,  # padded length
            1,  # labels dimension
        ]), sequence_weights_value.shape)

        self.assertAllEqual(np.array([
            [[1.0], [1.0]],
            [[1.0], [1.0]],
            [[1.0], [0.0]]  # Last time step masked
        ]), sequence_weights_value)

    def testProvidedUserWeightsActivations(self):
        user_weights = tf.constant([
            [[1.], [2.]],
            [[3.], [4.]],
            [[5.], [6.]],
        ])
        logits, length = dense_outputs()

        sequence_weights = make_sequence_weights(user_weights, logits, length)
        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            sequence_weights_value = sess.run(sequence_weights)

        self.assertAllEqual(np.array([
            3,  # batch size
            2,  # padded length
            1,  # labels dimension
        ]), sequence_weights_value.shape)

        self.assertAllEqual(np.array([
            [[1.0], [2.0]],
            [[3.0], [4.0]],
            [[5.0], [0.0]]  # Last time step masked
        ]), sequence_weights_value)
