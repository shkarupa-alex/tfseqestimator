from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.cudnn_rnn import CudnnLSTM, CudnnGRU
from tensorflow.contrib.rnn import LSTMBlockCell, GRUBlockCellV2, DropoutWrapper, MultiRNNCell
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn


class RnnImplementation:
    REGULAR = 'regular'
    CUDNN = 'cudnn'

    @classmethod
    def all(cls):
        return cls.REGULAR, cls.CUDNN

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            raise ValueError('Invalid RNN implementation: {}'.format(key))


class RnnDirection:
    UNIDIRECTIONAL = 'unidirectional'
    BIDIRECTIONAL = 'bidirectional'
    STACKED = 'stacked'

    @classmethod
    def all(cls):
        return cls.UNIDIRECTIONAL, cls.BIDIRECTIONAL, cls.STACKED

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            raise ValueError('Invalid RNN direction: {}'.format(key))


class RnnType:
    GRU = 'gru'
    LSTM = 'lstm'

    @classmethod
    def all(cls):
        return cls.GRU, cls.LSTM

    @classmethod
    def validate(cls, key):
        if key not in cls.all():
            raise ValueError('Invalid RNN cell: {}'.format(key))


def build_dynamic_rnn(sequence_input, sequence_length, params, is_training=False):
    """Build an RNN.

    Args:
      sequence_input: `Tensor` with shape `[batch_size, padded_length, d]` that will be passed as input to the RNN.
      sequence_length: `Tensor` with shape `[batch_size]` with real input sequences length.
      params: `HParams` instance with model parameters. Should contain:
          rnn_implementation: internal implementation. One of `RNNArchitecture` options.
          rnn_direction: layers direction. One of `RNNDirection` options.
            Stacked direction available only with regular implementation and 2+ layers.
          rnn_layers: number of layers.
          rnn_type: type of cell. One of `RNNCell` options.
          rnn_units: number of cells per layers.
          rnn_dropout: dropout rate, a number between [0, 1]. Applied after each layer.
            When set to 0 or None, dropout is disabled.
      is_training: whether this operation will be used in training or inference.

    Returns:
      rnn_outputs: `Tensor` with shape `[batch_size, padded_length, rnn_units * num_directions]` representing the
        output of all RNN time steps.
      last_output: `Tensor` with shape `[batch_size, rnn_units * num_directions]` representing the
        output of the last RNN time step.
    """

    with tf.name_scope('RNN'):
        # Convert to Time-major order
        sequence_input = tf.transpose(sequence_input, [1, 0, 2])

        # Create RNN
        RnnImplementation.validate(params.rnn_implementation)
        if RnnImplementation.REGULAR == params.rnn_implementation:
            rnn_outputs = _add_regular_rnn_layers(sequence_input, sequence_length, params, is_training)
        else:  # RnnImplementation.CUDNN == params.rnn_implementation
            rnn_outputs = _add_cudnn_rnn_layers(sequence_input, params, is_training)

        # Convert to Batch-major order
        rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])

        # Add final dropout
        if params.rnn_dropout:
            rnn_outputs = tf.layers.dropout(
                rnn_outputs,
                rate=params.rnn_dropout,
                training=is_training
            )

        # Extract last non-padded output
        last_output = _select_last_activations(rnn_outputs, sequence_length)

        return rnn_outputs, last_output


def _add_regular_rnn_layers(sequence_input, sequence_length, params, is_training=False):
    """Build a regular RNN.

    Args:
      sequence_input: `Tensor` with shape `[batch_size, padded_length, d]` that will be passed as input to the RNN.
      sequence_length: `Tensor` with shape `[batch_size]` with real input sequences length.
      params: `HParams` instance with model parameters. Should contain:
          rnn_direction: layers direction. One of `RNNDirection` options.
            Stacked direction available only with regular implementation and 2+ layers.
          rnn_layers: number of layers.
          rnn_type: type of cell. One of `RNNCell` options.
          rnn_units: number of cells per layers.
          rnn_dropout: dropout rate, a number between [0, 1]. Applied between layers.
            When set to 0 or None, dropout is disabled.
      is_training: whether this operation will be used in training or inference.

    Returns:
      `Tensor` with shape `[batch_size, padded_length, rnn_units * num_directions]` representing the output
        of the RNN time steps.
    """

    # Build forward cells
    cell_fw = _create_regular_rnn_cells(params, is_training)

    RnnDirection.validate(params.rnn_direction)
    if RnnDirection.UNIDIRECTIONAL == params.rnn_direction:
        # Build forward RNN
        rnn_outputs, _ = tf.nn.dynamic_rnn(
            cell=_combine_regular_rnn_cells(cell_fw),
            inputs=sequence_input,
            sequence_length=sequence_length,
            dtype=tf.float32,
            time_major=True
        )

        return rnn_outputs

    # Build backward cells
    cell_bw = _create_regular_rnn_cells(params, is_training=False)

    if RnnDirection.BIDIRECTIONAL == params.rnn_direction:
        # Build bi-direction RNN
        rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=_combine_regular_rnn_cells(cell_fw),
            cell_bw=_combine_regular_rnn_cells(cell_fw),
            inputs=sequence_input,
            sequence_length=sequence_length,
            dtype=tf.float32,
            time_major=True,
            # scope="rnn_classification" TODO
        )

        # Combine single-direction outputs Tensors by time axis
        rnn_outputs = tf.concat(rnn_outputs, 2)

    else:  # RnnDirection.STACKED == params.rnn_direction
        if 2 > params.rnn_layers:
            raise ValueError('Stacked RNN designed for 2+ layers. Use bidirectional RNN instead')

        # Build stacked bidirectional RNN
        rnn_outputs, _, _ = stack_bidirectional_dynamic_rnn(
            cells_fw=cell_fw,
            cells_bw=cell_bw,
            inputs=sequence_input,
            sequence_length=sequence_length,
            dtype=tf.float32,
            time_major=True,
        )

    return rnn_outputs


def _create_regular_rnn_cells(params, is_training=False):
    """Create regular RNN layers.

    Args:
      params: `HParams` instance with model parameters. Should contain:
          rnn_layers: number of layers.
          rnn_type: type of cell. One of `RNNCell` options.
          rnn_units: number of cells per layers.
          rnn_dropout: dropout rate, a number between [0, 1]. Applied between layers.
            When set to 0 or None, dropout is disabled.
      is_training: whether this operation will be used in training or inference.

    Returns:
      List of `RNNCell` instances.
    """

    RnnType.validate(params.rnn_type)
    if RnnType.GRU == params.rnn_type:
        regular_cell = GRUBlockCellV2
    else:  # RnnType.LSTM == params.rnn_type
        regular_cell = LSTMBlockCell

    # Build layers except last
    regular_layers = [regular_cell(params.rnn_units) for _ in range(params.rnn_layers - 1)]

    # Add dropout to each layer except last
    if is_training and params.rnn_dropout:
        regular_layers = [DropoutWrapper(cell, output_keep_prob=1.0 - params.rnn_dropout) for cell in regular_layers]

    # Add last layer without dropout for consistency with Cudnn
    regular_layers.append(regular_cell(params.rnn_units))

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


def _add_cudnn_rnn_layers(sequence_input, params, is_training=False):
    """Build a Cudnn RNN.

    Args:
      params: `HParams` instance with model parameters. Should contain:
          rnn_direction: layers direction. One of `RNNDirection` options.
            Stacked direction unavailable with Cudnn implementation.
          rnn_layers: number of layers.
          rnn_type: type of cell. One of `RNNCell` options.
          rnn_units: number of cells per layers.
          rnn_dropout: dropout rate, a number between [0, 1]. Applied between layers.
            When set to 0 or None, dropout is disabled.
      is_training: whether this operation will be used in training or inference.
      sequence_input: `Tensor` with shape `[batch_size, padded_length, d]` that will be passed as input to the RNN.

    Returns:
      `Tensor` with shape `[batch_size, padded_length, rnn_units * num_directions]` representing the output
        of the RNN time steps.
    """

    RnnType.validate(params.rnn_type)
    if RnnType.GRU == params.rnn_type:
        cudnn_cell = CudnnGRU
    else:  # RnnType.LSTM == params.rnn_type
        cudnn_cell = CudnnLSTM

    RnnDirection.validate(params.rnn_direction)
    if RnnDirection.UNIDIRECTIONAL == params.rnn_direction:
        cudnn_direction = 'unidirectional'
    else:  # RnnDirection.BIDIRECTIONAL == params.rnn_direction
        cudnn_direction = 'bidirectional'

    cudnn_dropout = params.rnn_dropout if is_training and params.rnn_dropout else 0.0

    # Build Cudnn RNN
    cudnn_rnn = cudnn_cell(
        num_layers=params.rnn_layers,
        num_units=params.rnn_units,
        direction=cudnn_direction,
        dropout=cudnn_dropout,
    )
    rnn_outputs, _ = cudnn_rnn(sequence_input, training=is_training)

    return rnn_outputs


def _select_last_activations(activations, sequence_lengths):
    """Selects the n-th set of activations for each n in `sequence_length`.
    Returns `Tensor` of shape `[batch_size, k]`.
    If `sequence_length` is not `None`, then `output[i, :] = activations[i, sequence_length[i] - 1, :]`.
    If `sequence_length` is `None`, then `output[i, :] = activations[i, -1, :]`.

    Args:
      activations: `Tensor` with shape `[batch_size, padded_length, k]`.
      sequence_lengths: `Tensor` with shape `[batch_size]` or `None`.

    Returns:
      `Tensor` of shape `[batch_size, k]`.
    """
    with tf.name_scope('select_last_activations', values=[activations, sequence_lengths]):
        batch_size, _, _ = tf.unstack(tf.shape(activations))
        batch_range = tf.range(batch_size)
        indices = tf.stack([batch_range, sequence_lengths - 1], axis=1)
        last_activations = tf.gather_nd(activations, indices)

        return last_activations
