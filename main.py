import drlearner as drl
from collections import deque
from multiprocessing import Pool
import random
import pickle
import time
import os
import tensorflow as tf
import numpy as np
from scipy.special import softmax as sp_softmax
import matplotlib.pyplot as plt

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def opponent(player):
    if player == 1:
        return 2
    else:
        return 1


def transform_state(state, player):
    t_state_1 = (state == player)
    t_state_2 = (state == opponent(player))

    return np.stack((t_state_1, t_state_2), axis=-1).astype(np.float32)


def mcts(game, model, n_mcts, d_alpha, d_epsilon, tau=0, tree=None, root_id=-1, training=False, verbose=False):
    if training:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(1)

    class Node:

        def __init__(self, node_id, parent_id, state, player, available_actions, result, back_value, policy):
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

            if int(np.sum(self.n) + 0.1) == 0:
                action = rng.choice([a for a, available in enumerate(self.available_actions) if available])
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
                policy.append(np.rot90(policy_temp.reshape(state.shape), -i).reshape((-1,)))
                value.append(v.numpy()[i][0])

            for i in range(4):
                policy_temp = p.numpy()[i + 4]
                policy.append(np.fliplr(np.rot90(policy_temp.reshape(state.shape), -i)).reshape((-1,)))
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
            state = game.update_state(tree[parent_id].state, action, tree[parent_id].player)
            result = game.update_result(state, action, tree[parent_id].player)
            player = game.update_player(state, tree[parent_id].player)
            available_actions = game.update_available_actions(state, player)
            policy, value = get_policy_value(state, player, available_actions)
            tree.append(Node(child_id, parent_id, state, player, available_actions, result, value, policy))
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

    policy, value = get_policy_value(game.state, game.player, game.available_actions, dirichlet_noise=training)

    if root_id < 0:
        tree = []
        root_id = 0
        parent_id = -1
        initial_result = -1
        tree.append(
            Node(root_id, parent_id, game.state, game.player, game.available_actions, initial_result, value, policy))
    else:
        tree[root_id].p = policy

    while tree[root_id].n.sum() < n_mcts:
        update_tree(tree, root_id)

    if verbose:
        print(policy.reshape(game.state.shape))
        print('--------------')
        print((tree[root_id].n / np.sum(tree[root_id].n)).reshape(game.state.shape))
        print('--------------')

    if tau == 0:
        p_mcts = np.zeros(shape=tree[root_id].n.shape)
        p_mcts[np.argmax(tree[root_id].n)] = 1
    else:
        p_mcts = tree[root_id].n ** (1 / tau) / np.sum(tree[root_id].n ** (1 / tau))

    action = rng.choice(len(game.available_actions), p=p_mcts)

    if not game.available_actions[action]:
        for a, available_action in enumerate(game.available_actions):
            if available_action:
                p_mcts = np.zeros(shape=tree[root_id].n.shape)
                p_mcts[a] = 1
                action = a + 0
                break

    root_id = tree[root_id].children_ids[action]

    return p_mcts, action, tree, root_id


def generate_episode_log(game, n_mcts, tau_turns, tau, d_alpha, d_epsilon):
    training_set = []
    game_logs = []
    tree = None
    root_id = -1
    turn = 0
    game.reset()

    while game.result == -1:
        turn += 1

        if turn > tau_turns:
            tau = 0

        p_mcts, action, tree, root_id = mcts(game, model, n_mcts, d_alpha, d_epsilon, tau, tree, root_id, training=True)
        game_logs.append((game.state, p_mcts, game.player))
        game.update(action)

    for state, p_mcts, player in game_logs:
        if game.result == player:
            z_reward = 1.0
        elif game.result == opponent(player):
            z_reward = -1.0
        else:
            z_reward = 0

        training_set.append((transform_state(state, player), p_mcts, z_reward))

        if game.h_flip:
            training_set.append((transform_state(np.fliplr(state), player),
                                 np.fliplr(p_mcts.reshape(state.shape)).reshape((-1,)), z_reward))

        if game.full_symmetry:
            for i in range(1, 4):
                training_set.append(
                    (transform_state(np.rot90(state, i), player),
                     np.rot90(p_mcts.reshape(state.shape), i).reshape((-1,)),
                     z_reward)
                )

            for i in range(4):
                training_set.append(
                    (transform_state(np.rot90(np.fliplr(state), i), player),
                     np.rot90(np.fliplr(p_mcts.reshape(state.shape)), i).reshape((-1,)),
                     z_reward)
                )

    return training_set


def init_child(game):
    global model
    model = tf.keras.models.load_model(os.path.join('objects', f'model_{game.name}_autosave.h5'))


def get_time(t):
    hours = int(t / 3600)
    t = t - 3600 * hours
    minutes = int(t / 60)
    t = t - 60 * minutes
    seconds = int(t)

    return hours, minutes, seconds


if __name__ == '__main__':

    game = drl.Reversi()
    mode = 'train'  # 'train', 'play', or 'watch'

    clear_replay_buffer = True
    generate_data = True

    initial_epoch = 0
    epochs = 10  # ? epochs per hour | Intel Core i7-860, 16GB DDR3 RAM, Nvidia GeForce GTX 1660 6GB

    batch_size = 512
    n_mcts = 200

    processes = 5  # Intel Core i7-860, 16GB DDR3 RAM, Nvidia GeForce GTX 1660 6GB
    episodes_per_epoch = 2 * processes
    epochs_per_checkpoint = 10
    replay_buffer_size = 100000
    training_steps_per_epoch = int(replay_buffer_size / batch_size) + 1
    c_l2 = 0.0001
    d_alpha = 10.0 / game.branching_factor
    d_epsilon = 0.25
    tau_turns = int(0.2 * game.avg_plies) + 1
    tau = 1

    if mode == 'train':

        if initial_epoch == 0:
            model = game.default_model()
            model.save(os.path.join('objects', f'model_{game.name}_autosave.h5'))
        else:
            model = tf.keras.models.load_model(os.path.join('objects', f'model_{game.name}_autosave.h5'))

        if clear_replay_buffer:
            replay_buffer = deque([], replay_buffer_size)
        else:
            replay_buffer = pickle.load(open(os.path.join('objects', f'replay_buffer_{game.name}_autosave.pkl'), 'rb'))
            replay_buffer = deque(replay_buffer, replay_buffer_size)

        with open(os.path.join('logs', f'model_summary_{game.name}.txt'), 'w') as model_summary:
            model.summary(print_fn=(lambda line: model_summary.write(f'{line}\n')))

        model.summary()

        optimizer = tf.keras.optimizers.Adam()
        loss_mse = tf.keras.losses.MeanSquaredError()
        l2_reg = tf.keras.regularizers.l2(c_l2)

        losses = []
        losses_policy = []
        losses_value = []
        losses_l2 = []
        start_time = time.time()

        for epoch in range(1, epochs + 1):

            epoch_losses = []
            epoch_losses_policy = []
            epoch_losses_value = []
            epoch_losses_l2 = []

            if generate_data:
                episode_args = [(game, n_mcts, tau_turns, tau, d_alpha, d_epsilon) for _ in
                                range(episodes_per_epoch)]

                with Pool(processes, initializer=init_child, initargs=(game,)) as pool:
                    pool_results = pool.starmap_async(generate_episode_log, episode_args)
                    episode_logs = pool_results.get()

                    for episode_log in episode_logs:
                        replay_buffer.extend(episode_log)

            for i in range(1, training_steps_per_epoch + 1):
                ds_batch = random.sample(replay_buffer, min(len(replay_buffer), batch_size))
                t_state, p_mcts, z_reward = map(list, zip(*ds_batch))

                with tf.GradientTape() as tape:
                    p_nn, v_nn = model(tf.stack(t_state), training=True)

                    ls_p_nn = tf.nn.log_softmax(p_nn)
                    loss_policy = -tf.math.reduce_mean(tf.math.reduce_sum(tf.math.multiply(p_mcts, ls_p_nn), axis=1))

                    loss_value = loss_mse(z_reward, v_nn)

                    weights_list = [tf.reshape(w_layer, (-1,)) for w_layer in model.trainable_variables]
                    weights = tf.concat(weights_list, axis=0)
                    loss_l2 = l2_reg(weights)

                    loss = loss_policy + loss_value + loss_l2

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

                epoch_losses.append(loss)
                epoch_losses_policy.append(loss_policy)
                epoch_losses_value.append(loss_value)
                epoch_losses_l2.append(loss_l2)

            if (epoch + initial_epoch) % epochs_per_checkpoint == 0:
                model.save(os.path.join('objects', f'model_{game.name}_{epoch + initial_epoch:04d}.h5'))
                pickle.dump(
                    replay_buffer,
                    open(os.path.join('objects', f'replay_buffer_{game.name}_{epoch + initial_epoch:04d}.pkl'), 'wb'))

            model.save(os.path.join('objects', f'model_{game.name}_autosave.h5'))
            pickle.dump(replay_buffer, open(os.path.join('objects', f'replay_buffer_{game.name}_autosave.pkl'), 'wb'))

            losses.extend(epoch_losses)
            losses_policy.extend(epoch_losses_policy)
            losses_value.extend(epoch_losses_value)
            losses_l2.extend(epoch_losses_l2)

            seconds_per_epoch = (time.time() - start_time) / epoch
            epochs_remaining = epochs - epoch
            seconds_remaining = seconds_per_epoch * epochs_remaining
            elapsed_time = get_time(time.time() - start_time)
            time_remaining = get_time(seconds_remaining)

            print(f'Epoch: {epoch:6d}/{epochs} | '
                  f'Remaining: {time_remaining[0]:2d}:{time_remaining[1]:02d}:{time_remaining[2]:02d} | '
                  f'Elapsed: {elapsed_time[0]:2d}:{elapsed_time[1]:02d}:{elapsed_time[2]:02d} | '
                  f'Loss: {np.mean(epoch_losses):.4f} (Policy: {np.mean(epoch_losses_policy):.4f}, '
                  f'Value: {np.mean(epoch_losses_value):.4f}, L2: {np.mean(epoch_losses_l2):.4f}) | '
                  f'Replay buffer size: {len(replay_buffer):6d}')

        fig = plt.figure(figsize=(6, 4), dpi=600)
        ax = fig.add_subplot(1, 1, 1)
        alpha = 0.75
        initial_step = initial_epoch * training_steps_per_epoch + 1
        plt.plot([i + initial_step for i, _ in enumerate(losses)], losses, label='Total', alpha=alpha)
        plt.plot([i + initial_step for i, _ in enumerate(losses_policy)], losses_policy, label='Policy', alpha=alpha)
        plt.plot([i + initial_step for i, _ in enumerate(losses_value)], losses_value, label='Value', alpha=alpha)
        plt.plot([i + initial_step for i, _ in enumerate(losses_l2)], losses_l2, label='L2', alpha=alpha)
        plt.legend()
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss')
        plt.tight_layout()
        plt.savefig(os.path.join('logs', f'learning_curve_{game.name}_{initial_epoch:04d}.png'))

    if mode in ('play', 'watch'):
        model = tf.keras.models.load_model(os.path.join('objects', f'model_{game.name}_autosave.h5'))

        game.reset()
        print(game.state)
        print('--------------')

        while game.result is -1:

            if game.player is 1:

                if mode is 'play':
                    actions_table = [i for i in range(game.available_actions.size)]
                    actions_table = np.array(actions_table) * game.available_actions
                    actions_table = actions_table.reshape(game.state.shape)
                    print(actions_table)
                    action = int(input("Choose a tile: "))

                else:
                    _, action, _, _ = mcts(game, model, n_mcts, d_alpha, d_epsilon, verbose=True)
            else:
                _, action, _, _ = mcts(game, model, n_mcts, d_alpha, d_epsilon, verbose=True)

            game.update(action)

            print(game.state)
            print('--------------')

        print(f'Result: {game.result}')
