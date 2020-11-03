import tensorflow as tf
import numpy as np
from scipy.special import softmax as sp_softmax


def opponent(player):
    if player == 1:
        return 2
    else:
        return 1


def transform_state(state, player):
    t_state_1 = state == player
    t_state_2 = state == opponent(player)

    return np.stack((t_state_1, t_state_2), axis=-1).astype(np.float32)


def mcts(
    game, model, n_mcts, tau=0, tree=None, root_id=-1, training=False, mp_threshold=1
):
    d_alpha = 10.0 / game.branching_factor
    d_epsilon = 0.25

    rng = np.random.default_rng()

    class Node:
        def __init__(
            self,
            node_id,
            parent_id,
            state,
            player,
            available_actions,
            result,
            back_value,
            policy,
        ):
            self.node_id = node_id
            self.parent_id = parent_id
            self.children_ids = [-1] * policy.shape[0]
            self.state = state
            self.player = player
            self.available_actions = available_actions
            self.opponent = 2 if player == 1 else 1
            self.result = result
            self.back_value = back_value
            self.c_puct = 1.0

            self.n = np.zeros(policy.shape)
            self.w = np.zeros(policy.shape)
            self.q = np.zeros(policy.shape)
            self.p = policy

        def mcts_action(self):

            u = self.c_puct * self.p * (np.sqrt(np.sum(self.n)) / (1 + self.n))

            next_search_proba = self.q + u
            min_proba = np.amin(next_search_proba) - 1
            for a, available_action in enumerate(self.available_actions):
                if not available_action:
                    next_search_proba[a] = min_proba

            if np.sum(self.n) < 0.5:
                action = np.argmax(self.p)
            else:
                action = np.argmax(next_search_proba)

            return action

        def update_node(self, action, value):
            self.n[action] += 1
            self.w[action] += value
            self.q[action] = self.w[action] / self.n[action]
            self.back_value = value

    def get_policy_value(state, player, available_actions, dirichlet_noise=False):

        t_state = [transform_state(state, player)]

        if game.h_flip:
            t_state.append(transform_state(np.fliplr(state), player))

        if game.full_symmetry:
            for i in range(1, 4):
                t_state.append(transform_state(np.rot90(state, i), player))

            for i in range(4):
                t_state.append(transform_state(np.rot90(np.fliplr(state), i), player))

        p, v = model(tf.convert_to_tensor(t_state), training=training)
        policy = p.numpy()[0]
        value = v.numpy()[0][0]

        if game.h_flip:
            policy_fliplr = p.numpy()[1]
            policy_fliplr = np.fliplr(policy_fliplr.reshape(state.shape)).reshape((-1,))
            value_fliplr = v.numpy()[1][0]
            policy = (policy + policy_fliplr) / 2
            value = (value_fliplr + value_fliplr) / 2

        if game.full_symmetry:
            policy = [policy]
            value = [value]

            for i in range(1, 4):
                policy_temp = p.numpy()[i]
                policy.append(
                    np.rot90(policy_temp.reshape(state.shape), -i).reshape((-1,))
                )
                value.append(v.numpy()[i][0])

            for i in range(4):
                policy_temp = p.numpy()[i + 4]
                policy.append(
                    np.fliplr(np.rot90(policy_temp.reshape(state.shape), -i)).reshape(
                        (-1,)
                    )
                )
                value.append(v.numpy()[i + 4][0])

            policy = np.mean(policy, axis=0)
            value = np.mean(value)

        for i, available in enumerate(available_actions):
            if not available:
                policy[i] = -10000

        policy = sp_softmax(policy)
        policy = policy * available_actions

        if dirichlet_noise:
            n_pool = rng.dirichlet([d_alpha] * int(available_actions.sum() + 0.1))
            n = np.zeros_like(policy)
            n_i = 0
            for i, available in enumerate(available_actions):
                if available:
                    n[i] = n_pool[n_i]
                    n_i += 1

            policy = (1 - d_epsilon) * policy + d_epsilon * n

        if policy.sum() > 0:
            policy = policy / policy.sum()

        return policy, value

    def update_tree(tree, parent_id):
        action = tree[parent_id].mcts_action()
        child_id = tree[parent_id].children_ids[action]

        if child_id == -1:
            child_id = len(tree)
            state = game.update_state(
                tree[parent_id].state, action, tree[parent_id].player
            )
            result = game.update_result(state, action, tree[parent_id].player)
            player = game.update_player(state, tree[parent_id].player)
            available_actions = game.update_available_actions(state, player)
            policy, value = get_policy_value(state, player, available_actions)
            tree.append(
                Node(
                    child_id,
                    parent_id,
                    state,
                    player,
                    available_actions,
                    result,
                    value,
                    policy,
                )
            )
            tree[parent_id].children_ids[action] = child_id

            if tree[child_id].result == tree[parent_id].player:
                parent_value = 1
            elif tree[child_id].result == tree[parent_id].opponent:
                parent_value = -1
            elif tree[child_id].result == 0:
                parent_value = 0
            elif tree[child_id].player == tree[parent_id].opponent:
                parent_value = -tree[child_id].back_value
            else:
                parent_value = tree[child_id].back_value

            tree[parent_id].update_node(action, parent_value)

        else:
            if tree[child_id].result == tree[parent_id].player:
                parent_value = 1
            elif tree[child_id].result == tree[parent_id].opponent:
                parent_value = -1
            elif tree[child_id].result == 0:
                parent_value = 0
            else:
                update_tree(tree, child_id)
                if tree[child_id].player == tree[parent_id].opponent:
                    parent_value = -tree[child_id].back_value
                else:
                    parent_value = tree[child_id].back_value

            tree[parent_id].update_node(action, parent_value)

    policy, value = get_policy_value(
        game.state, game.player, game.available_actions, dirichlet_noise=training
    )

    if not training:
        print("------ Policy ------")
        print(100 * policy.reshape(game.state.shape) // 1)
        print("--------------------")

    if n_mcts == 0:
        return -1, np.argmax(policy), -1, -1

    if root_id < 0:
        tree = []
        root_id = 0
        parent_id = -1
        initial_result = -1
        tree.append(
            Node(
                root_id,
                parent_id,
                game.state,
                game.player,
                game.available_actions,
                initial_result,
                value,
                policy,
            )
        )
    else:
        tree[root_id].p = policy

    searches = 0
    while tree[root_id].n.sum() < n_mcts:
        update_tree(tree, root_id)
        searches += 1

    if not training:

        policy_max = np.amax(policy)
        proba_max = mp_threshold * policy_max
        proba_min = 1 / np.sum(game.available_actions)
        policy_action = np.argmax(policy)

        mcts_results = tree[root_id].n / np.sum(tree[root_id].n)
        mp_ratio = mcts_results[policy_action] / policy_max

        if proba_min < mcts_results[policy_action] < proba_max:
            print(f"MCTS / Policy: {mp_ratio:.0%} | Searches: {searches}")
            print("--------------------")
            print("------- MCTS -------")
            print(100 * mcts_results.reshape(game.state.shape) // 1)
            print("--------------------")

        while (
            tree[root_id].n.sum() < 800
            and proba_min < mcts_results[policy_action] < proba_max
        ):
            update_tree(tree, root_id)
            mcts_results = tree[root_id].n / np.sum(tree[root_id].n)
            searches += 1

        mcts_results = tree[root_id].n / np.sum(tree[root_id].n)
        mp_ratio = mcts_results[policy_action] / policy_max
        print(f"MCTS / Policy: {mp_ratio:.0%} | Searches: {searches}")
        print("--------------------")
        print("------- MCTS -------")
        print(100 * mcts_results.reshape(game.state.shape) // 1)
        print("--------------------")

    mcts_optimal = np.argmax(tree[root_id].n)
    p_mcts = np.zeros(tree[root_id].n.shape)
    p_mcts[mcts_optimal] = 1

    if tau < 0.5:
        action = mcts_optimal
    else:
        proba = tree[root_id].n ** (1 / tau) / np.sum(tree[root_id].n ** (1 / tau))
        action = rng.choice(len(game.available_actions), p=proba)
        if not game.available_actions[action]:
            action = mcts_optimal

    root_id = tree[root_id].children_ids[action]

    return p_mcts, action, tree, root_id


def generate_episode_log(game, model_path, n_mcts, tau_initial):
    model = tf.keras.models.load_model(model_path)

    tau_decay = np.log(1 / tau_initial) / (-0.5)
    training_set = []
    game_logs = []
    tree = None
    root_id = -1
    turn = 0
    tau = 1
    game.reset()

    while game.result == -1:
        turn += 1
        tau = tau_initial * np.exp(-tau_decay * turn / game.avg_plies)

        p_mcts, action, tree, root_id = mcts(
            game, model, n_mcts, tau, tree, root_id, training=True
        )
        game_logs.append((game.state, p_mcts, game.player))
        game.update(action)

    for i, (state, p_mcts, player) in enumerate(game_logs, 1):

        if game.result == player:
            z_reward = 1
        elif game.result == opponent(player):
            z_reward = -1
        else:
            z_reward = 0

        training_set.append((transform_state(state, player), p_mcts, z_reward))

        if game.h_flip:
            training_set.append(
                (
                    transform_state(np.fliplr(state), player),
                    np.fliplr(p_mcts.reshape(state.shape)).reshape((-1,)),
                    z_reward,
                )
            )

        if game.full_symmetry:
            for i in range(1, 4):
                training_set.append(
                    (
                        transform_state(np.rot90(state, i), player),
                        np.rot90(p_mcts.reshape(state.shape), i).reshape((-1,)),
                        z_reward,
                    )
                )

            for i in range(4):
                training_set.append(
                    (
                        transform_state(np.rot90(np.fliplr(state), i), player),
                        np.rot90(np.fliplr(p_mcts.reshape(state.shape)), i).reshape(
                            (-1,)
                        ),
                        z_reward,
                    )
                )

    return training_set


def alpha_zero_model(
    input_shape, n_actions, residual_blocks=8, filters=64, kernel_size=3
):
    """ AlphaZero used: residual_blocks=19, filters=256 """

    def residual_block(input_layer):
        res_block_out = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")(
            input_layer
        )
        res_block_out = tf.keras.layers.BatchNormalization()(res_block_out)
        res_block_out = tf.keras.layers.LeakyReLU()(res_block_out)
        res_block_out = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")(
            res_block_out
        )
        res_block_out = tf.keras.layers.BatchNormalization()(res_block_out)
        res_block_out = tf.keras.layers.Add()([res_block_out, input_layer])
        res_block_out = tf.keras.layers.LeakyReLU()(res_block_out)

        return res_block_out

    def policy_head(input_layer):
        policy_out = tf.keras.layers.Conv2D(filters=2, kernel_size=1)(input_layer)
        policy_out = tf.keras.layers.BatchNormalization()(policy_out)
        policy_out = tf.keras.layers.LeakyReLU()(policy_out)
        policy_out = tf.keras.layers.Flatten()(policy_out)
        policy_out = tf.keras.layers.Dense(n_actions, name="policy")(policy_out)

        return policy_out

    def value_head(input_layer):
        value_out = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(input_layer)
        value_out = tf.keras.layers.BatchNormalization()(value_out)
        value_out = tf.keras.layers.LeakyReLU()(value_out)
        value_out = tf.keras.layers.Flatten()(value_out)
        value_out = tf.keras.layers.Dense(filters)(value_out)
        value_out = tf.keras.layers.LeakyReLU()(value_out)
        value_out = tf.keras.layers.Dense(1, activation="tanh", name="value")(value_out)

        return value_out

    inputs = tf.keras.Input(shape=input_shape)
    conv_block_out = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")(
        inputs
    )
    conv_block_out = tf.keras.layers.BatchNormalization()(conv_block_out)
    tower_out = tf.keras.layers.LeakyReLU()(conv_block_out)

    for _ in range(residual_blocks):
        tower_out = residual_block(tower_out)

    policy_outputs = policy_head(tower_out)
    value_outputs = value_head(tower_out)

    model = tf.keras.Model(inputs=inputs, outputs=[policy_outputs, value_outputs])

    return model


class ConnectFour:
    name = "connect-four"

    branching_factor = 4
    avg_plies = 36

    symmetry_factor = 2

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

        return alpha_zero_model(
            input_shape=(self._ROWS, self._COLS, self._CHANNELS),
            n_actions=self._ACTIONS,
        )

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

        row_a, col_a = self.get_row_col(action)
        state_p = state == player

        # Check vertical
        if np.sum(state_p[row_a : row_a + self._CONNECT, col_a]) == self._CONNECT:
            return player

        # Check horizontal
        for j in range(self._COLS - self._CONNECT + 1):
            if np.sum(state_p[row_a, j : j + self._CONNECT]) == self._CONNECT:
                return player

        # Check \ diagonal
        diagonal_nw_se = np.diag(state_p, col_a - row_a)
        if np.sum(diagonal_nw_se) >= self._CONNECT:
            for k in range(diagonal_nw_se.size - self._CONNECT + 1):
                if np.sum(diagonal_nw_se[k : k + self._CONNECT]) == self._CONNECT:
                    return player

        # Check / diagonal
        col_i = (self._COLS - 1) - col_a
        diagonal_ne_sw = np.diag(state_p[:, ::-1], col_i - row_a)
        if np.sum(diagonal_ne_sw) >= self._CONNECT:
            for k in range(diagonal_ne_sw.size - self._CONNECT + 1):
                if np.sum(diagonal_ne_sw[k : k + self._CONNECT]) == self._CONNECT:
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


class Reversi:
    name = "reversi"

    branching_factor = 10
    avg_plies = 58

    symmetry_factor = 8

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
        return alpha_zero_model(
            input_shape=(self._ROWS, self._COLS, self._CHANNELS),
            n_actions=self._ACTIONS,
        )

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
