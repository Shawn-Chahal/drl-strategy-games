import drlearner as drl

from multiprocessing import Pool
import random
import pickle
import time
import os

import tensorflow as tf
import numpy as np
import pandas as pd
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

    game = drl.Reversi()
    mode = 'train'  # 'train' or 'play'
    plot_only = False
    initial_epoch = 348
    final_epoch = 1000

    PROCESSES = 5
    BATCH_SIZE = 512
    convergence_threshold = 0.01
    convergence_factor = 2
    max_iterations_per_frame = 100
    n_mcts_max = 800
    C_L2 = 0.0001

    model_path = os.path.join('objects', f'model_{game.name}_{initial_epoch:04d}.h5')

    if mode is 'train':

        if initial_epoch == 0:
            model = game.default_model()
            model.save(model_path)
            dict_loss = {'Iteration': [], 'n_mcts': [], 'Loss (Total)': [], 'Loss (Policy)': [],
                         'Loss (Value)': [], 'Loss (L2)': [], 'Epoch': []}
            dict_time = {'Epoch': [0], 'Time [s]': [0.0]}
            optimizer = tf.keras.optimizers.Adam()

        else:
            model = tf.keras.models.load_model(model_path)
            optimizer = pickle.load(open(os.path.join('objects', f'optimizer_{game.name}.pkl'), 'rb'))

            dict_loss = pd.read_csv(os.path.join('logs', f'loss_{game.name}.csv')).to_dict('list')
            dict_time = pd.read_csv(os.path.join('logs', f'time_{game.name}.csv')).to_dict('list')

        with open(os.path.join('logs', f'model_summary_{game.name}.txt'), 'w') as model_summary:
            model.summary(print_fn=(lambda line: model_summary.write(f'{line}\n')))

        model.summary()
        loss_mse = tf.keras.losses.MeanSquaredError()
        l2_reg = tf.keras.regularizers.l2(C_L2)
        start_time = time.time()

        if not plot_only:

            for epoch in range(initial_epoch + 1, final_epoch + 1):

                n_mcts = min(n_mcts_max, epoch)

                with Pool(PROCESSES) as pool:
                    episode_args = [(game, model_path, n_mcts) for _ in range(PROCESSES)]
                    pool_results = pool.starmap_async(drl.generate_episode_log, episode_args)
                    episode_logs = pool_results.get()
                    training_data = []
                    for episode_log in episode_logs:
                        training_data.extend(episode_log)

                n_batches = int(len(training_data) / BATCH_SIZE + 1)
                convergence_iterations = convergence_factor * n_batches
                for iteration in range(max_iterations_per_frame * n_batches):
                    ds_batch = random.sample(training_data, min(len(training_data), BATCH_SIZE))
                    t_state, p_mcts, z_reward = map(list, zip(*ds_batch))

                    with tf.GradientTape() as tape:
                        p_nn, v_nn = model(tf.stack(t_state), training=True)

                        ls_p_nn = tf.nn.log_softmax(p_nn)
                        loss_policy = -tf.math.reduce_mean(
                            tf.math.reduce_sum(tf.math.multiply(p_mcts, ls_p_nn), axis=1))

                        loss_value = loss_mse(z_reward, v_nn)

                        weights_list = [tf.reshape(w_layer, (-1,)) for w_layer in model.trainable_variables]
                        weights = tf.concat(weights_list, axis=0)
                        loss_l2 = l2_reg(weights)

                        loss = loss_policy + loss_value + loss_l2

                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

                    dict_loss['n_mcts'].append(n_mcts)
                    dict_loss['Loss (Total)'].append(loss.numpy())
                    dict_loss['Loss (Policy)'].append(loss_policy.numpy())
                    dict_loss['Loss (Value)'].append(loss_value.numpy())
                    dict_loss['Loss (L2)'].append(loss_l2.numpy())
                    dict_loss['Iteration'].append(len(dict_loss['Loss (Total)']))
                    dict_loss['Epoch'].append(epoch)

                    print(f'Iteration: {dict_loss["Iteration"][-1]} | '
                          f'Loss: {loss.numpy():.4f} (Policy: {loss_policy.numpy():.4f}, '
                          f'Value: {loss_value.numpy():.4f}, L2: {loss_l2.numpy():.4f}) | '
                          f'Replay buffer size: {len(training_data)}')

                    if iteration > convergence_iterations:
                        delta_convergence = (dict_loss['Loss (Total)'][-convergence_iterations] -
                                             dict_loss['Loss (Total)'][-1])
                        if delta_convergence < convergence_threshold:
                            break

                model_path = os.path.join('objects', f'model_{game.name}_{epoch:04d}.h5')
                model.save(model_path)
                model.save(os.path.join('best-networks', f'model_{game.name}.h5'))
                pickle.dump(optimizer, open(os.path.join('objects', f'optimizer_{game.name}.pkl'), 'wb'))

                elapsed_time = time.time() - start_time
                elapsed_time_r = get_time(elapsed_time)

                total_time = elapsed_time + dict_time['Time [s]'][initial_epoch]
                total_time_r = get_time(total_time)

                dict_time['Epoch'].append(epoch)
                dict_time['Time [s]'].append(total_time)

                pd.DataFrame.from_dict(dict_loss).to_csv(os.path.join('logs', f'loss_{game.name}.csv'), index=False)
                pd.DataFrame.from_dict(dict_time).to_csv(os.path.join('logs', f'time_{game.name}.csv'), index=False)

                print(f'Epoch: {epoch}/{final_epoch} | '
                      f'Elapsed: {elapsed_time_r[0]}:{elapsed_time_r[1]:02d}:{elapsed_time_r[2]:02d} | '
                      f'Total: {total_time_r[0]}:{total_time_r[1]:02d}:{total_time_r[2]:02d}')

        alpha = 0.75
        n_rows = 3
        n_cols = 2

        fig = plt.figure(figsize=(8, 6), dpi=600)

        ax = fig.add_subplot(n_rows, n_cols, 1)
        ax.plot(dict_loss['Iteration'], dict_loss['Loss (Total)'], label='Total', alpha=alpha)
        ax.plot(dict_loss['Iteration'], dict_loss['Loss (Policy)'], label='Policy', alpha=alpha)
        ax.plot(dict_loss['Iteration'], dict_loss['Loss (Value)'], label='Value', alpha=alpha)
        ax.plot(dict_loss['Iteration'], dict_loss['Loss (L2)'], label='L2', alpha=alpha)
        ax.legend(ncol=2)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')

        ax = fig.add_subplot(n_rows, n_cols, 3)
        ax.plot(dict_loss['Iteration'], dict_loss['n_mcts'], label='MCTS', alpha=alpha)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Number of MCTS searches')

        ax = fig.add_subplot(n_rows, n_cols, 5)
        ax.plot(np.array(dict_time['Time [s]']) / 3600, dict_time['Epoch'])
        ax.set_xlabel('Time [h]')
        ax.set_ylabel('Epoch')

        for i, loss_type in enumerate(('Policy', 'Value', 'L2'), 1):
            ax = fig.add_subplot(n_rows, n_cols, 2 * i)
            ax.plot(dict_loss['Iteration'], dict_loss[f'Loss ({loss_type})'])
            ax.set_xlabel('Iterations')
            ax.set_ylabel(f'Loss ({loss_type})')

        plt.tight_layout()
        plt.savefig(os.path.join('logs', f'learning_curve_{game.name}.png'))

    if mode is 'play':
        model = tf.keras.models.load_model(os.path.join('best-networks', f'model_{game.name}.h5'))
        n_mcts = n_mcts_max + 0
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
