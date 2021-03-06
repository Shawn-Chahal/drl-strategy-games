import drlearner as drl

from multiprocessing import Pool
from collections import deque
import random
import pickle
import time
import os

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def learning_curve(width, height):
    fig = plt.figure(figsize=(width, height), dpi=600)

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(dict_loss["Epoch"], dict_loss["Loss (Total)"], label="Total")
    ax.plot(dict_loss["Epoch"], dict_loss["Loss (Policy)"], label="Policy")
    ax.plot(dict_loss["Epoch"], dict_loss["Loss (Value)"], label="Value")
    ax.plot(dict_loss["Epoch"], dict_loss["Loss (L2)"], label="L2")
    ax.legend(ncol=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig(os.path.join("logs", f"learning_curve_{game.name}.png"))
    plt.close(fig)


def dashboard(width, height, n_last_epochs=None):
    n_rows = 1
    n_cols = 4

    if n_last_epochs is None:
        start = 0
    else:
        start = -min(n_last_epochs, epoch)

    fig = plt.figure(figsize=(width, height), dpi=600)

    ax = fig.add_subplot(n_rows, n_cols, 1)
    ax.plot(
        dict_loss["Epoch"][start:], dict_loss["Loss (Total)"][start:], color="tab:blue"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Total)")

    ax = fig.add_subplot(n_rows, n_cols, 2)
    ax.plot(
        dict_loss["Epoch"][start:],
        dict_loss[f"Loss (Policy)"][start:],
        color="tab:orange",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Loss (Policy)")

    ax = fig.add_subplot(n_rows, n_cols, 3)
    ax.plot(
        dict_loss["Epoch"][start:],
        dict_loss[f"Loss (Value)"][start:],
        color="tab:green",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Loss (Value)")

    ax = fig.add_subplot(n_rows, n_cols, 4)
    ax.plot(
        dict_loss["Epoch"][start:], dict_loss[f"Loss (L2)"][start:], color="tab:red"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Loss (L2)")

    plt.tight_layout()

    if n_last_epochs is None:
        plt.savefig(os.path.join("logs", f"learning_curve_{game.name}_dashboard.png"))
    else:
        plt.savefig(
            os.path.join("logs", f"learning_curve_{game.name}_dashboard_recent.png")
        )
    plt.close(fig)


def get_time(t):
    hours = int(t / 3600)
    t = t - 3600 * hours
    minutes = int(t / 60)
    t = t - 60 * minutes
    seconds = int(t)

    return hours, minutes, seconds


if __name__ == "__main__":

    game = drl.Reversi()
    mode = "train"  # "train" or "play"
    initial_epoch = 313
    final_epoch = 1000

    PROCESSES = 5
    BATCH_SIZE = 512
    N_MCTS = 800
    C_L2 = 0.0001

    tau_initial = 2
    deque_growth = 1 / 3

    frames_per_epoch = PROCESSES * game.avg_plies * game.symmetry_factor
    model_path = os.path.join("objects", f"model_{game.name}_{initial_epoch:04d}.h5")

    if mode is "train":

        if initial_epoch == 0:
            model = game.default_model()
            model.save(model_path)
            dict_loss = {
                "Epoch": [],
                "Loss (Total)": [],
                "Loss (Policy)": [],
                "Loss (Value)": [],
                "Loss (L2)": [],
                "Time [s]": [],
            }
            replay_buffer = deque([], frames_per_epoch)
            optimizer = tf.keras.optimizers.Adam()

        else:
            model = tf.keras.models.load_model(model_path)
            optimizer = pickle.load(
                open(os.path.join("objects", f"optimizer_{game.name}.pkl"), "rb")
            )
            replay_buffer = pickle.load(
                open(os.path.join("objects", f"replay_buffer_{game.name}.pkl"), "rb")
            )
            dict_loss = pd.read_csv(
                os.path.join("logs", f"loss_{game.name}.csv")
            ).to_dict("list")

        with open(
                os.path.join("logs", f"model_summary_{game.name}.txt"), "w"
        ) as model_summary:
            model.summary(print_fn=(lambda line: model_summary.write(f"{line}\n")))

        model.summary()
        loss_mse = tf.keras.losses.MeanSquaredError()
        l2_reg = tf.keras.regularizers.l2(C_L2)
        start_time = time.time()

        for epoch in range(initial_epoch + 1, final_epoch + 1):
            replay_buffer = deque(
                replay_buffer,
                int(deque_growth * epoch * frames_per_epoch),
            )
            episode_args = [
                (game, model_path, N_MCTS, tau_initial) for _ in range(PROCESSES)
            ]

            with Pool(PROCESSES) as pool:
                pool_results = pool.starmap_async(
                    drl.generate_episode_log, episode_args
                )
                episode_logs = pool_results.get()

            for episode_log in episode_logs:
                replay_buffer.extend(episode_log)

            replay_buffer_list = list(replay_buffer)
            random.shuffle(replay_buffer_list)
            ds = [
                replay_buffer_list[i: i + BATCH_SIZE]
                for i in range(0, len(replay_buffer_list), BATCH_SIZE)
            ]
            epoch_loss = {"Total": [], "Policy": [], "Value": [], "L2": []}

            for ds_batch in ds:
                t_state, p_mcts, z_reward = map(list, zip(*ds_batch))

                with tf.GradientTape() as tape:
                    p_nn, v_nn = model(tf.stack(t_state), training=True)

                    ls_p_nn = tf.nn.log_softmax(p_nn)
                    loss_policy = -tf.math.reduce_mean(
                        tf.math.reduce_sum(tf.math.multiply(p_mcts, ls_p_nn), axis=1)
                    )

                    loss_value = loss_mse(z_reward, v_nn)

                    weights_list = [
                        tf.reshape(w_layer, (-1,))
                        for w_layer in model.trainable_variables
                    ]
                    weights = tf.concat(weights_list, axis=0)
                    loss_l2 = l2_reg(weights)

                    loss = loss_policy + loss_value + loss_l2

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                    grads_and_vars=zip(grads, model.trainable_variables)
                )

                epoch_loss["Total"].append(loss.numpy())
                epoch_loss["Policy"].append(loss_policy.numpy())
                epoch_loss["Value"].append(loss_value.numpy())
                epoch_loss["L2"].append(loss_l2.numpy())

            dict_loss["Loss (Total)"].append(np.mean(np.array(epoch_loss["Total"])))
            dict_loss["Loss (Policy)"].append(np.mean(np.array(epoch_loss["Policy"])))
            dict_loss["Loss (Value)"].append(np.mean(np.array(epoch_loss["Value"])))
            dict_loss["Loss (L2)"].append(np.mean(np.array(epoch_loss["L2"])))
            dict_loss["Epoch"].append(epoch)

            model_path = os.path.join("objects", f"model_{game.name}_{epoch:04d}.h5")
            model.save(model_path)
            pickle.dump(
                optimizer,
                open(os.path.join("objects", f"optimizer_{game.name}.pkl"), "wb"),
            )

            pickle.dump(
                replay_buffer,
                open(os.path.join("objects", f"replay_buffer_{game.name}.pkl"), "wb"),
            )
            elapsed_time = time.time() - start_time
            elapsed_time_r = get_time(elapsed_time)
            if initial_epoch is 0:
                total_time = elapsed_time
            else:
                total_time = elapsed_time + dict_loss["Time [s]"][initial_epoch - 1]
            total_time_r = get_time(total_time)
            dict_loss["Time [s]"].append(total_time)
            pd.DataFrame.from_dict(dict_loss).to_csv(
                os.path.join("logs", f"loss_{game.name}.csv"), index=False
            )

            epochs_per_day = epoch / total_time * 3600 * 24
            timestamp = time.strftime("%H:%M:%S", time.localtime())

            print(
                f"Epoch: {epoch}/{final_epoch} | "
                f"Time: {total_time_r[0]}:{total_time_r[1]:02d}:{total_time_r[2]:02d} | "
                f'Loss: {dict_loss["Loss (Total)"][-1]:.4f} (Policy: {dict_loss["Loss (Policy)"][-1]:.4f}, '
                f'Value: {dict_loss["Loss (Value)"][-1]:.4f}, L2: {dict_loss["Loss (L2)"][-1]:.4f}) | '
                f"Replay buffer size: {len(replay_buffer)} | Epochs/day: {epochs_per_day:.0f} | "
                f"Timestamp: {timestamp}"
            )

            learning_curve(6, 4)
            dashboard(16, 4)
            dashboard(16, 4, int(epochs_per_day + 1))

    if mode is "play":
        model = tf.keras.models.load_model(
            os.path.join("objects", f"model_{game.name}_{initial_epoch:04d}.h5")
        )
        game.reset()
        player_human = int(input("Choose a player [1, 2]: "))
        print(game.state)
        print("--------------")

        while game.result is -1:

            if game.player is player_human:
                actions_table = [i for i in range(game.available_actions.size)]
                actions_table = np.array(actions_table) * game.available_actions
                actions_table = actions_table.reshape(game.state.shape)
                print(actions_table)
                action = int(input("Choose a tile: "))

            else:
                _, action, _, _ = drl.mcts(game, model, 0)

            game.update(action)

            print(game.state)
            print("--------------")

        print(f"Result: {game.result}")
