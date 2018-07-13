from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from ..rnn import build_dynamic_rnn, select_last_activations
from ..dense import DenseActivation, build_logits_activations, apply_time_distributed
from .test_rnn import rnn_params, sequence_outputs


def rnn_outputs():
    sequence_output, sequence_length = sequence_outputs()
    rnn_output = build_dynamic_rnn(
        sequence_output,
        sequence_length,
        rnn_params(),
    )
    last_output = select_last_activations(rnn_output, sequence_length)

    return rnn_output, last_output


def dense_params():
    params = rnn_params()
    params.add_hparam('dense_layers', [7, 6])
    params.add_hparam('dense_activation', DenseActivation.RELU)
    params.add_hparam('dense_dropout', 0.0)

    return params


class BuildLogitsActivationsTest(tf.test.TestCase):
    def testClassificationLogitsShape(self):
        rnn_output, last_output = rnn_outputs()
        dense_logits = apply_time_distributed(
            build_logits_activations,
            rnn_output,
            dense_params().override_from_dict({
                'dense_layers': [],
            }),
            logits_size=5
        )
        last_logit = build_logits_activations(
            last_output,
            dense_params().override_from_dict({
                'dense_layers': [],
            }),
            logits_size=5
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
            build_logits_activations,
            rnn_output,
            dense_params(),
            logits_size=1
        )
        last_logit = build_logits_activations(
            last_output,
            dense_params(),
            logits_size=1
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
