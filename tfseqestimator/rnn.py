from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.cudnn_rnn import CudnnLSTM, CudnnGRU
from tensorflow.contrib.rnn import LSTMBlockCell, GRUBlockCellV2, DropoutWrapper, MultiRNNCell
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn


class RnnType:
    REGULAR_FORWARD_GRU = 'regular_forward_gru'
    REGULAR_FORWARD_LSTM = 'regular_forward_lstm'

    REGULAR_BIDIRECTIONAL_GRU = 'regular_bidirectional_gru'
    REGULAR_BIDIRECTIONAL_LSTM = 'regular_bidirectional_lstm'

    REGULAR_STACKED_GRU = 'regular_stacked_gru'
    REGULAR_STACKED_LSTM = 'regular_stacked_lstm'

    CUDNN_FORWARD_GRU = 'cudnn_forward_gru'
    CUDNN_FORWARD_LSTM = 'cudnn_forward_lstm'

    CUDNN_BIDIRECTIONAL_GRU = 'cudnn_bidirectional_gru'
    CUDNN_BIDIRECTIONAL_LSTM = 'cudnn_bidirectional_lstm'

    @classmethod
    def all(cls):
        return (
            cls.REGULAR_FORWARD_GRU, cls.REGULAR_FORWARD_LSTM,
            cls.REGULAR_BIDIRECTIONAL_GRU, cls.REGULAR_BIDIRECTIONAL_LSTM,
            cls.REGULAR_STACKED_GRU, cls.REGULAR_STACKED_LSTM,
            cls.CUDNN_FORWARD_GRU, cls.CUDNN_FORWARD_LSTM,
            cls.CUDNN_BIDIRECTIONAL_GRU, cls.CUDNN_BIDIRECTIONAL_LSTM
        )

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            raise ValueError('Invalid RNN type: {}'.format(key))

    @classmethod
    def is_regular(cls, key):
        return key in (
            cls.REGULAR_FORWARD_GRU, cls.REGULAR_FORWARD_LSTM,
            cls.REGULAR_BIDIRECTIONAL_GRU, cls.REGULAR_BIDIRECTIONAL_LSTM,
            cls.REGULAR_STACKED_GRU, cls.REGULAR_STACKED_LSTM,
        )

    @classmethod
    def is_cudnn(cls, key):
        return key in (
            cls.CUDNN_FORWARD_GRU, cls.CUDNN_FORWARD_LSTM,
            cls.CUDNN_BIDIRECTIONAL_GRU, cls.CUDNN_BIDIRECTIONAL_LSTM
        )

    @classmethod
    def is_forward(cls, key):
        return key in (
            cls.REGULAR_FORWARD_GRU, cls.REGULAR_FORWARD_LSTM,
            cls.CUDNN_FORWARD_GRU, cls.CUDNN_FORWARD_LSTM
        )

    @classmethod
    def is_bidirectional(cls, key):
        return key in (
            cls.REGULAR_BIDIRECTIONAL_GRU, cls.REGULAR_BIDIRECTIONAL_LSTM,
            cls.CUDNN_BIDIRECTIONAL_GRU, cls.CUDNN_BIDIRECTIONAL_LSTM
        )

    @classmethod
    def is_stacked(cls, key):
        return key in (
            cls.REGULAR_STACKED_GRU, cls.REGULAR_STACKED_LSTM,
        )

    @classmethod
    def is_lstm(cls, key):
        return key in (
            cls.REGULAR_FORWARD_LSTM, cls.REGULAR_BIDIRECTIONAL_LSTM, cls.REGULAR_STACKED_LSTM,
            cls.CUDNN_FORWARD_LSTM, cls.CUDNN_BIDIRECTIONAL_LSTM
        )

    @classmethod
    def is_gru(cls, key):
        return key in (
            cls.REGULAR_FORWARD_GRU, cls.REGULAR_BIDIRECTIONAL_GRU, cls.REGULAR_STACKED_GRU,
            cls.CUDNN_FORWARD_GRU, cls.CUDNN_BIDIRECTIONAL_GRU
        )


def build_dynamic_rnn(sequence_input, sequence_length, rnn_type, rnn_layers, rnn_dropout, is_training):
    """Build an RNN.

    Args:
      sequence_input: `Tensor` with shape `[batch_size, padded_length, ?]` that will be passed as input to the RNN.
      sequence_length: `Tensor` with shape `[batch_size]` with actual input sequences length.
      rnn_type: type, direction and implementations of RNN. One of `RnnType` options.
      rnn_layers: iterable of integer number of hidden units per layer.
      rnn_dropout: recurrent layers dropout rate, a number between [0, 1]. Applied after each layer.
        When set to 0 or None, dropout is disabled.
      is_training: whether this operation will be used in training or inference.

    Returns:
      `Tensor` with shape `[batch_size, padded_length, rnn_units * num_directions]` representing the
        output of all RNN time steps.
    """

    with tf.variable_scope('dynamic_rnn') as rnn_scope:
        RnnType.validate(rnn_type)

        if not len(rnn_layers):
            raise ValueError('At least one layer required for RNN.')

        # Convert to Time-major order
        sequence_input = tf.transpose(sequence_input, [1, 0, 2], name='time_major')

        # Create RNN
        if RnnType.is_regular(rnn_type):
            rnn_outputs = _add_regular_rnn_layers(
                sequence_input=sequence_input,
                sequence_length=sequence_length,
                rnn_type=rnn_type,
                rnn_layers=rnn_layers,
                rnn_dropout=rnn_dropout,
                is_training=is_training,
            )
        else:
            assert RnnType.is_cudnn(rnn_type)
            rnn_outputs = _add_cudnn_rnn_layers(
                sequence_input=sequence_input,
                rnn_type=rnn_type,
                rnn_layers=rnn_layers,
                rnn_dropout=rnn_dropout,
                is_training=is_training,
            )

        # Convert to Batch-major order
        rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2], name='batch_major')

        # Add final dropout
        if rnn_dropout:
            rnn_outputs = tf.layers.dropout(
                rnn_outputs,
                rate=rnn_dropout,
                training=is_training,
            )

        return rnn_outputs


def _add_regular_rnn_layers(sequence_input, sequence_length, rnn_type, rnn_layers, rnn_dropout, is_training):
    """Build a regular RNN.

    Args:
      sequence_input: `Tensor` with shape `[batch_size, padded_length, ?]` that will be passed as input to the RNN.
      sequence_length: `Tensor` with shape `[batch_size]` with actual input sequences length.
      rnn_type: type, direction and implementations of RNN. One of `RnnType` options.
      rnn_layers: iterable of integer number of hidden units per layer.
      rnn_dropout: recurrent layers dropout rate, a number between [0, 1]. Applied after each layer.
        When set to 0 or None, dropout is disabled.
      is_training: whether this operation will be used in training or inference.

    Returns:
      `Tensor` with shape `[batch_size, padded_length, rnn_units * num_directions]` representing the output of all
        RNN time steps.
    """

    assert RnnType.is_regular(rnn_type)

    # with tf.name_scope('forward_rnn') as forward_scope:
    # Forward cells
    cell_fw = _create_regular_rnn_cells(
        rnn_type=rnn_type,
        rnn_layers=rnn_layers,
        rnn_dropout=rnn_dropout,
        is_training=is_training,
    )

    if RnnType.is_forward(rnn_type):
        rnn_outputs, _ = tf.nn.dynamic_rnn(
            cell=_combine_regular_rnn_cells(cell_fw),
            inputs=sequence_input,
            sequence_length=sequence_length,
            dtype=tf.float32,
            time_major=True,
        )

        return rnn_outputs

    # Backward cells
    cell_bw = _create_regular_rnn_cells(
        rnn_type=rnn_type,
        rnn_layers=rnn_layers,
        rnn_dropout=rnn_dropout,
        is_training=is_training,
    )

    if RnnType.is_bidirectional(rnn_type):
        rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=_combine_regular_rnn_cells(cell_fw),
            cell_bw=_combine_regular_rnn_cells(cell_fw),
            inputs=sequence_input,
            sequence_length=sequence_length,
            dtype=tf.float32,
            time_major=True,
        )

        # Combine single-direction outputs Tensors by time axis
        rnn_outputs = tf.concat(rnn_outputs, 2)

    else:
        assert RnnType.is_stacked(rnn_type)
        if 1 == len(rnn_layers):
            tf.logging.warning('Stacked RNN designed for 2 and more layers. Consider using bidirectional RNN instead.')

        rnn_outputs, _, _ = stack_bidirectional_dynamic_rnn(
            cells_fw=cell_fw,
            cells_bw=cell_bw,
            inputs=sequence_input,
            sequence_length=sequence_length,
            dtype=tf.float32,
            time_major=True,
        )

    return rnn_outputs


def _create_regular_rnn_cells(rnn_type, rnn_layers, rnn_dropout, is_training):
    """Create regular RNN layers.

    Args:
      rnn_type: type, direction and implementations of RNN. One of `RnnType` options.
      rnn_layers: iterable of integer number of hidden units per layer.
      rnn_dropout: recurrent layers dropout rate, a number between [0, 1]. Applied after each layer.
        When set to 0 or None, dropout is disabled.
      is_training: whether this operation will be used in training or inference.

    Returns:
      List of `RNNCell` instances.
    """

    if RnnType.is_gru(rnn_type):
        regular_cell = GRUBlockCellV2
    else:
        assert RnnType.is_lstm(rnn_type)
        regular_cell = LSTMBlockCell

    # Build layers except last
    regular_layers = []
    for num_units in rnn_layers[:-1]:
        cell = regular_cell(num_units)

        if is_training and rnn_dropout:
            # Add dropout to each layer except last
            cell = DropoutWrapper(
                cell,
                output_keep_prob=1.0 - rnn_dropout,
                variational_recurrent=True  # Should be better choice https://arxiv.org/pdf/1707.09861.pdf
            )

        regular_layers.append(cell)

    # Add last layer without dropout for consistency with Cudnn
    regular_layers.append(regular_cell(rnn_layers[-1]))

    return regular_layers


def _combine_regular_rnn_cells(rnn_cells):
    """Combine multiple `RNNCell`s to single one

    Args:
      rnn_cells: list of `RNNCell` instances.

    Returns:
      `RNNCell` instance.
    """
    if 1 == len(rnn_cells):
        # Return single layer if possible
        return rnn_cells[0]
    else:
        # Otherwise combine and return multiple layers
        return MultiRNNCell(rnn_cells)


def _add_cudnn_rnn_layers(sequence_input, rnn_type, rnn_layers, rnn_dropout, is_training):
    """Build a Cudnn RNN.

    Args:
      sequence_input: `Tensor` with shape `[batch_size, padded_length, ?]` that will be passed as input to the RNN.
      rnn_type: type of cell. One of `RNNCell` options.
      rnn_layers: number of layers.
      rnn_dropout: dropout rate, a number between [0, 1]. Applied between layers.
        When set to 0 or None, dropout is disabled.
      is_training: whether this operation will be used in training or inference.

    Returns:
      `Tensor` with shape `[batch_size, padded_length, rnn_units * num_directions]` representing the output of all
        RNN time steps.
    """

    assert RnnType.is_cudnn(rnn_type)

    if 1 != len(set(rnn_layers)):
        tf.logging.warning('Cudnn RNNs does not support different layers sizes. Maximum size will be used.')
    cudnn_layers = len(rnn_layers)
    cuddnn_units = max(rnn_layers)

    if RnnType.is_gru(rnn_type):
        cudnn_cell = CudnnGRU
    else:
        assert RnnType.is_lstm(rnn_type)
        cudnn_cell = CudnnLSTM

    if RnnType.is_forward(rnn_type):
        cudnn_direction = 'unidirectional'
    else:
        assert RnnType.is_bidirectional(rnn_type)
        cudnn_direction = 'bidirectional'

    cudnn_dropout = rnn_dropout if is_training and rnn_dropout else 0.0

    with tf.device('/gpu:0'):  # CuDnn requires to be pinned to single device
        partitioner = tf.get_variable_scope().partitioner
        tf.get_variable_scope().set_partitioner(None)  # Partitioner not supported with CuDnn

        # Build Cudnn RNN
        cudnn_rnn = cudnn_cell(
            num_layers=cudnn_layers,
            num_units=cuddnn_units,
            direction=cudnn_direction,
            dropout=cudnn_dropout,
        )
        rnn_outputs, _ = cudnn_rnn(sequence_input, training=is_training)

        tf.get_variable_scope().set_partitioner(partitioner)

    return rnn_outputs


def select_last_activations(rnn_outputs, sequence_length):
    """Selects the n-th set of activations for each n in `_get_sequence_length`.
    Returns `Tensor` of shape `[batch_size, k]`.
    `output[i, :] = activations[i, _get_sequence_length[i] - 1, :]`.

    Args:
      rnn_outputs: `Tensor` with shape `[batch_size, padded_length, k]`.
      sequence_length: `Tensor` with shape `[batch_size]`.

    Returns:
      `Tensor` of shape `[batch_size, k]`.
    """
    with tf.name_scope('last_output'):
        batch_size, _, _ = tf.unstack(tf.shape(rnn_outputs, out_type=tf.int64))
        batch_range = tf.range(batch_size, dtype=tf.int64)
        indices = tf.stack([batch_range, sequence_length - 1], axis=1)
        last_activations = tf.gather_nd(rnn_outputs, indices)

        return last_activations
