import tensorflow as tf
import numpy as np


def alpha_zero_model(input_shape, n_actions, residual_blocks=19, filters=256, kernel_size=3):
    def residual_block(input_layer):
        res_block_out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(input_layer)
        res_block_out = tf.keras.layers.BatchNormalization()(res_block_out)
        res_block_out = tf.keras.layers.LeakyReLU()(res_block_out)
        res_block_out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(res_block_out)
        res_block_out = tf.keras.layers.BatchNormalization()(res_block_out)
        res_block_out = tf.keras.layers.Add()([res_block_out, input_layer])
        res_block_out = tf.keras.layers.LeakyReLU()(res_block_out)

        return res_block_out

    def policy_head(input_layer):
        policy_out = tf.keras.layers.Conv2D(filters=2, kernel_size=1)(input_layer)
        policy_out = tf.keras.layers.BatchNormalization()(policy_out)
        policy_out = tf.keras.layers.LeakyReLU()(policy_out)
        policy_out = tf.keras.layers.Flatten()(policy_out)
        policy_out = tf.keras.layers.Dense(n_actions, name='policy')(policy_out)

        return policy_out

    def value_head(input_layer):
        value_out = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(input_layer)
        value_out = tf.keras.layers.BatchNormalization()(value_out)
        value_out = tf.keras.layers.LeakyReLU()(value_out)
        value_out = tf.keras.layers.Flatten()(value_out)
        value_out = tf.keras.layers.Dense(filters)(value_out)
        value_out = tf.keras.layers.LeakyReLU()(value_out)
        value_out = tf.keras.layers.Dense(1, activation='tanh', name='value')(value_out)

        return value_out

    inputs = tf.keras.Input(shape=input_shape)
    conv_block_out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(inputs)
    conv_block_out = tf.keras.layers.BatchNormalization()(conv_block_out)
    tower_out = tf.keras.layers.LeakyReLU()(conv_block_out)

    for _ in range(residual_blocks):
        tower_out = residual_block(tower_out)

    policy_outputs = policy_head(tower_out)
    value_outputs = value_head(tower_out)

    model = tf.keras.Model(inputs=inputs, outputs=[policy_outputs, value_outputs])

    return model


class ConnectFour:
    name = 'connect-four'

    branching_factor = 4
    avg_plies = 36
    symmetry_factor = 2

    h_flip = True

    _ROWS = 6
    _COLS = 7
    _ACTIONS = _ROWS * _COLS
    _CHANNELS = 2
    _RESIDUAL_BLOCKS = 7
    _FILTERS = 64
    _CONNECT = 4

    def __init__(self):
        self.reset()

    def reset(self):
        self.initialize_state()
        self.result = -1
        self.player = 1
        self.available_actions = self.update_available_actions(self.state, self.player)

    def default_model(self):

        return alpha_zero_model(input_shape=(self._ROWS, self._COLS, self._CHANNELS), n_actions=self._ACTIONS,
                                residual_blocks=self._RESIDUAL_BLOCKS, filters=self._FILTERS)

    def initialize_state(self):
        self.state = np.zeros((self._ROWS, self._COLS))

    def get_row_col(self, action):
        row = action // self._COLS
        col = action % self._COLS
        return row, col

    def update_state(self, state, action, player):
        state = state.copy()
        row, col = self.get_row_col(action)
        state[row, col] = player
        return state

    def update_result(self, state, action, player):

        for i in range(self._ROWS):
            for j in range(self._COLS):

                # Check vertical
                if i <= self._ROWS - self._CONNECT:
                    if sum([1 for k in range(self._CONNECT) if state[i + k, j] == player]) == self._CONNECT:
                        return player

                # Check horizontal
                if j <= self._COLS - self._CONNECT:
                    if sum([1 for k in range(self._CONNECT) if state[i, j + k] == player]) == self._CONNECT:
                        return player

                # Check \ diagonal
                if (i <= self._ROWS - self._CONNECT) and (j <= self._COLS - self._CONNECT):
                    if sum([1 for k in range(self._CONNECT) if
                            state[i + k, j + k] == player]) == self._CONNECT:
                        return player

                # Check / diagonal
                if (i <= self._ROWS - self._CONNECT) and (j >= self._CONNECT - 1):
                    if sum([1 for k in range(self._CONNECT) if
                            state[i + k, j - k] == player]) == self._CONNECT:
                        return player

        if (state == 0).sum() == 0:
            return 0
        else:
            return -1

    def update_player(self, player):
        if player == 1:
            return 2
        else:
            return 1

    def update_available_actions(self, state, player):
        available_actions = np.zeros(state.shape)

        for j in range(self._COLS):
            for i in range(self._ROWS - 1, -1, -1):
                if state[i, j] == 0:
                    available_actions[i, j] = True
                    break

        return available_actions.reshape((-1,))

    def update(self, action):
        self.state = self.update_state(self.state, action, self.player)
        self.result = self.update_result(self.state, action, self.player)
        self.player = self.update_player(self.player)
        self.available_actions = self.update_available_actions(self.state, self.player)
