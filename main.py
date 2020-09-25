import drlearner as drl

from collections import deque
from multiprocessing import Pool
import random
import pickle
import time
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_time(t):
    hours = int(t / 3600)
    t = t - 3600 * hours
    minutes = int(t / 60)
    t = t - 60 * minutes
    seconds = int(t)

    return hours, minutes, seconds


if __name__ == '__main__':

    game = drl.ConnectFour()
    mode = 'train'  # 'train' or 'play'

    initial_epoch = 630
    epochs = 4

    """ Training hyperparameters """
    replay_buffer_episodes = 3000
    replay_buffer_refresh = 0.02

    """ System-dependant hyperparameters """
    processes = 5
    batch_size = 512

    """ Fixed hyperparameters """
    n_mcts = 800
    c_l2 = 0.0001
    tau_turns = int(0.2 * game.avg_plies) + 1
    replay_buffer_size = replay_buffer_episodes * game.avg_plies
    episodes_per_epoch = int(replay_buffer_refresh * replay_buffer_episodes / game.symmetry_factor) + 1
    training_steps_per_epoch = int(replay_buffer_size / batch_size) + 1

    model_path = os.path.join('objects', f'model_{game.name}_{initial_epoch:04d}.h5')

    if mode is 'train':

        if initial_epoch == 0:
            model = game.default_model()
            model.save(model_path)
            replay_buffer = deque([], replay_buffer_size)
        else:
            model = tf.keras.models.load_model(model_path)
            replay_buffer = pickle.load(
                open(os.path.join('objects', f'replay_buffer_{game.name}_{initial_epoch:04d}.pkl'), 'rb'))
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

            episode_args = [(game, model_path, n_mcts, tau_turns) for _ in range(episodes_per_epoch)]

            with Pool(processes) as pool:
                pool_results = pool.starmap_async(drl.generate_episode_log, episode_args)
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

            model_path = os.path.join('objects', f'model_{game.name}_{epoch + initial_epoch:04d}.h5')
            model.save(model_path)
            pickle.dump(
                replay_buffer,
                open(os.path.join('objects', f'replay_buffer_{game.name}_{epoch + initial_epoch:04d}.pkl'), 'wb'))

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

    if mode is 'play':
        model = tf.keras.models.load_model(model_path)

        game.reset()
        print(game.state)
        print('--------------')

        while game.result is -1:

            if game.player is 1:

                actions_table = [i for i in range(game.available_actions.size)]
                actions_table = np.array(actions_table) * game.available_actions
                actions_table = actions_table.reshape(game.state.shape)
                print(actions_table)
                action = int(input("Choose a tile: "))

            else:
                _, action, _, _ = drl.mcts(game, model, n_mcts, verbose=True)

            game.update(action)

            print(game.state)
            print('--------------')

        print(f'Result: {game.result}')
