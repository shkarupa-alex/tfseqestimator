from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from ..rnn import build_dynamic_rnn, select_last_activations
from ..dense import build_logits_activations, apply_time_distributed, DenseActivation
from .test_rnn import RnnType, sequence_outputs


def rnn_outputs():
    sequence_output, sequence_length = sequence_outputs()
    rnn_output = build_dynamic_rnn(
        sequence_input=sequence_output,
        sequence_length=sequence_length,
        rnn_type=RnnType.REGULAR_FORWARD_GRU,
        rnn_layers=[8],
        rnn_dropout=0.,
        is_training=False,
    )
    last_output = select_last_activations(rnn_output, sequence_length)

    return rnn_output, last_output


class BuildLogitsActivationsTest(tf.test.TestCase):
    def testClassificationLogitsShape(self):
        rnn_output, last_output = rnn_outputs()
        dense_logits = apply_time_distributed(
            layer_producer=build_logits_activations,
            sequence_input=rnn_output,
            logits_size=5,
            dense_layers=[-3, 2],
            dense_activation=tf.nn.sigmoid,
            dense_dropout=0.,
            dense_norm=False,
            is_training=False,
        )
        last_logit = build_logits_activations(
            flat_input=last_output,
            logits_size=5,
            dense_layers=[5],
            dense_activation=tf.nn.sigmoid,
            dense_dropout=0.,
            dense_norm=False,
            is_training=False,
        )

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            dense_logits_value, last_logit_value = sess.run([dense_logits, last_logit])

        self.assertAllEqual(np.array([
            3,  # expected batch size
            2,  # padded sequence length
            5,  # expected logits count
        ]), dense_logits_value.shape)

        self.assertAllEqual(np.array([
            3,  # expected batch size
            5,  # expected logits count
        ]), last_logit_value.shape)

    def testRegressionLogitsShape(self):
        rnn_output, last_output = rnn_outputs()
        dense_logits = apply_time_distributed(
            layer_producer=build_logits_activations,
            sequence_input=rnn_output,
            logits_size=1,
            dense_layers=[-5, 7, 6],
            dense_activation=DenseActivation.TANH,
            dense_dropout=0.,
            dense_norm=True,
            is_training=False,
        )
        last_logit = build_logits_activations(
            flat_input=last_output,
            logits_size=1,
            dense_layers=[7, 6],
            dense_activation=tf.nn.sigmoid,
            dense_dropout=0.,
            dense_norm=True,
            is_training=False,
        )

        with self.test_session() as sess:
            sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
            dense_logits_value, last_logit_value = sess.run([dense_logits, last_logit])

        self.assertAllEqual(np.array([
            3,  # expected batch size
            2,  # padded sequence length
            1,  # expected logits count
        ]), dense_logits_value.shape)

        self.assertAllEqual(np.array([
            3,  # expected batch size
            1,  # expected logits count
        ]), last_logit_value.shape)
