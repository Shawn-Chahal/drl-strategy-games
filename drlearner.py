import tensorflow as tf
import numpy as np


def alpha_zero_model(input_shape, n_actions, residual_blocks=8, filters=64, kernel_size=3):
    """ AlphaZero used: residual_blocks=19, filters=256 """

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


class Reversi:
    name = 'reversi'

    branching_factor = 10
    avg_plies = 58

    h_flip = False
    full_symmetry = True

    _ROWS = 8
    _COLS = _ROWS + 0
    _ACTIONS = _ROWS * _COLS
    _CHANNELS = 2

    def __init__(self):
        self.reset()

    def reset(self):
        self.initialize_state()
        self.result = -1
        self.player = 1
        self.available_actions = self.update_available_actions(self.state, self.player)

    def default_model(self):
        return alpha_zero_model(input_shape=(self._ROWS, self._COLS, self._CHANNELS), n_actions=self._ACTIONS)

    def initialize_state(self):
        self.state = np.zeros((self._ROWS, self._COLS))

        self.state[self._ROWS // 2 - 1, self._COLS // 2 - 1] = 2
        self.state[self._ROWS // 2 - 1, self._COLS // 2] = 1
        self.state[self._ROWS // 2, self._COLS // 2 - 1] = 1
        self.state[self._ROWS // 2, self._COLS // 2] = 2

    def get_row_col(self, action):
        row = action // self._COLS
        col = action % self._COLS
        return row, col

    def update_state(self, state, action, player):
        state = state.copy()
        row, col = self.get_row_col(action)
        state[row, col] = player

        link = np.zeros(8)

        if player == 1:
            opponent = 2
        else:
            opponent = 1

        # Check right [0]
        if col < self._COLS - 2:
            if state[row][col + 1] == opponent:
                for k in range(2, self._COLS - col):
                    if state[row][col + k] == 0:
                        break
                    elif state[row][col + k] == player:
                        link[0] = True
                        break

        # Check up-right [1]
        if (row > 1) and (col < self._COLS - 2):
            if state[row - 1][col + 1] == opponent:
                for k in range(2, min(row + 1, self._COLS - col)):
                    if state[row - k][col + k] == 0:
                        break
                    elif state[row - k][col + k] == player:
                        link[1] = True
                        break

        # Check up [2]
        if row > 1:
            if state[row - 1][col] == opponent:
                for k in range(2, row + 1):
                    if state[row - k][col] == 0:
                        break
                    elif state[row - k][col] == player:
                        link[2] = True
                        break

        # Check up-left [3]
        if (row > 1) and (col > 1):
            if state[row - 1][col - 1] == opponent:
                for k in range(2, min(row + 1, col + 1)):
                    if state[row - k][col - k] == 0:
                        break
                    elif state[row - k][col - k] == player:
                        link[3] = True
                        break

        # Check left [4]
        if col > 1:
            if state[row][col - 1] == opponent:
                for k in range(2, col + 1):
                    if state[row][col - k] == 0:
                        break
                    elif state[row][col - k] == player:
                        link[4] = True
                        break

        # Check down-left [5]
        if (row < self._ROWS - 2) and (col > 1):
            if state[row + 1][col - 1] == opponent:
                for k in range(2, min(self._ROWS - row, col + 1)):
                    if state[row + k][col - k] == 0:
                        break
                    elif state[row + k][col - k] == player:
                        link[5] = True
                        break

        # Check down [6]
        if row < self._ROWS - 2:
            if state[row + 1][col] == opponent:
                for k in range(2, self._ROWS - row):
                    if state[row + k][col] == 0:
                        break
                    elif state[row + k][col] == player:
                        link[6] = True
                        break

        # Check down-right [7]
        if (row < self._ROWS - 2) and (col < self._COLS - 2):
            if state[row + 1][col + 1] == opponent:
                for k in range(2, min(self._ROWS - row, self._COLS - col)):
                    if state[row + k][col + k] == 0:
                        break
                    elif state[row + k][col + k] == player:
                        link[7] = True
                        break

        # Check right [0]
        if link[0]:
            for k in range(1, self._COLS - col):
                if state[row][col + k] == player:
                    break
                else:
                    state[row][col + k] = player

        # Check up-right [1]
        if link[1]:
            for k in range(1, min(row + 1, self._COLS - col)):
                if state[row - k][col + k] == player:
                    break
                else:
                    state[row - k][col + k] = player

        # Check up [2]
        if link[2]:
            for k in range(1, row + 1):
                if state[row - k][col] == player:
                    break
                else:
                    state[row - k][col] = player

        # Check up-left [3]
        if link[3]:
            for k in range(1, min(row + 1, col + 1)):
                if state[row - k][col - k] == player:
                    break
                else:
                    state[row - k][col - k] = player

        # Check left [4]
        if link[4]:
            for k in range(1, col + 1):
                if state[row][col - k] == player:
                    break
                else:
                    state[row][col - k] = player

        # Check down-left [5]
        if link[5]:
            for k in range(1, min(self._ROWS - row, col + 1)):
                if state[row + k][col - k] == player:
                    break
                else:
                    state[row + k][col - k] = player

        # Check down [6]
        if link[6]:
            for k in range(1, self._ROWS - row):
                if state[row + k][col] == player:
                    break
                else:
                    state[row + k][col] = player

        # Check down-right [7]
        if link[7]:
            for k in range(1, min(self._ROWS - row, self._COLS - col)):
                if state[row + k][col + k] == player:
                    break
                else:
                    state[row + k][col + k] = player

        return state

    def update_result(self, state, action, player):

        for i in (1, 2):
            if self.update_available_actions(state, i).sum() > 0:
                return -1

        p1_score = (state == 1).sum()
        p2_score = (state == 2).sum()

        if p1_score > p2_score:
            return 1
        elif p2_score > p1_score:
            return 2
        else:
            return 0

    def update_player(self, state, player):
        if player == 1:
            next_player = 2
        else:
            next_player = 1

        next_available_actions = self.update_available_actions(state, next_player)

        for available in next_available_actions:
            if available:
                return next_player

        return player

    def update_available_actions(self, state, player):

        def action_available(state, action):
            state = state.copy()
            row, col = self.get_row_col(action)

            state[row][col] = player

            if player == 1:
                opponent = 2
            else:
                opponent = 1

            # Check right [0]
            if col < self._COLS - 2:
                if state[row][col + 1] == opponent:
                    for k in range(2, self._COLS - col):
                        if state[row][col + k] == 0:
                            break
                        elif state[row][col + k] == player:
                            return True

            # Check up-right [1]
            if (row > 1) and (col < self._COLS - 2):
                if state[row - 1][col + 1] == opponent:
                    for k in range(2, min(row + 1, self._COLS - col)):
                        if state[row - k][col + k] == 0:
                            break
                        elif state[row - k][col + k] == player:
                            return True

            # Check up [2]
            if row > 1:
                if state[row - 1][col] == opponent:
                    for k in range(2, row + 1):
                        if state[row - k][col] == 0:
                            break
                        elif state[row - k][col] == player:
                            return True

            # Check up-left [3]
            if (row > 1) and (col > 1):
                if state[row - 1][col - 1] == opponent:
                    for k in range(2, min(row + 1, col + 1)):
                        if state[row - k][col - k] == 0:
                            break
                        elif state[row - k][col - k] == player:
                            return True

            # Check left [4]
            if col > 1:
                if state[row][col - 1] == opponent:
                    for k in range(2, col + 1):
                        if state[row][col - k] == 0:
                            break
                        elif state[row][col - k] == player:
                            return True

            # Check down-left [5]
            if (row < self._ROWS - 2) and (col > 1):
                if state[row + 1][col - 1] == opponent:
                    for k in range(2, min(self._ROWS - row, col + 1)):
                        if state[row + k][col - k] == 0:
                            break
                        elif state[row + k][col - k] == player:
                            return True

            # Check down [6]
            if row < self._ROWS - 2:
                if state[row + 1][col] == opponent:
                    for k in range(2, self._ROWS - row):
                        if state[row + k][col] == 0:
                            break
                        elif state[row + k][col] == player:
                            return True

            # Check down-right [7]
            if (row < self._ROWS - 2) and (col < self._COLS - 2):
                if state[row + 1][col + 1] == opponent:
                    for k in range(2, min(self._ROWS - row, self._COLS - col)):
                        if state[row + k][col + k] == 0:
                            break
                        elif state[row + k][col + k] == player:
                            return True

            return False

        available_actions = np.zeros(self._ACTIONS)

        for a in range(self._ACTIONS):
            row, col = self.get_row_col(a)
            if state[row, col] == 0:
                available_actions[a] = action_available(state, a)

        return available_actions

    def update(self, action):
        self.state = self.update_state(self.state, action, self.player)
        self.result = self.update_result(self.state, action, self.player)
        self.player = self.update_player(self.state, self.player)
        self.available_actions = self.update_available_actions(self.state, self.player)


class ConnectFour:
    name = 'connect-four'

    branching_factor = 4
    avg_plies = 36

    h_flip = True
    full_symmetry = False

    _ROWS = 6
    _COLS = 7
    _ACTIONS = _ROWS * _COLS
    _CHANNELS = 2
    _CONNECT = 4

    def __init__(self):
        self.reset()

    def reset(self):
        self.initialize_state()
        self.result = -1
        self.player = 1
        self.available_actions = self.update_available_actions(self.state, self.player)

    def default_model(self):

        return alpha_zero_model(input_shape=(self._ROWS, self._COLS, self._CHANNELS), n_actions=self._ACTIONS)

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

    def update_player(self, state, player):
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
        self.player = self.update_player(self.state, self.player)
        self.available_actions = self.update_available_actions(self.state, self.player)
