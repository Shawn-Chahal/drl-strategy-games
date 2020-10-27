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


def tournament(game, model_1_epoch, model_2_epoch, n_mcts_play, match_number):
    print(f"Match: {match_number}")

    model_1 = tf.keras.models.load_model(
        os.path.join("objects", f"model_{game.name}_{model_1_epoch:04d}.h5")
    )
    model_2 = tf.keras.models.load_model(
        os.path.join("objects", f"model_{game.name}_{model_2_epoch:04d}.h5")
    )
    game.reset()

    while game.result is -1:
        if game.player is 1:
            _, action, _, _ = drl.mcts(game, model_1, n_mcts_play)
        else:
            _, action, _, _ = drl.mcts(game, model_2, n_mcts_play)

        game.update(action)

    return model_1_epoch, model_2_epoch, game.result


def get_time(t):
    hours = int(t / 3600)
    t = t - 3600 * hours
    minutes = int(t / 60)
    t = t - 60 * minutes
    seconds = int(t)

    return hours, minutes, seconds


if __name__ == "__main__":

    PROCESSES = 5
    BATCH_SIZE = 512
    N_MCTS = 800
    C_L2 = 0.0001

    game = drl.ConnectFour()
    mode = "train"  # 'train', 'play', or 'tournament'
    initial_epoch = 416
    final_epoch = 1000

    dashboard_epochs = 50
    n_mcts_play = 200
    tournament_range = [
        i for i in range(0, initial_epoch + 1, max(initial_epoch // 10, 1))
    ]
    frames_per_epoch = PROCESSES * game.avg_plies * game.symmetry_factor
    min_deque_size = 10 * frames_per_epoch
    deque_growth = 0.4
    tau_turns = int(0.2 * game.avg_plies + 1)

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
            replay_buffer = deque([], min_deque_size)
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
                max(min_deque_size, int(deque_growth * epoch * frames_per_epoch)),
            )
            episode_args = [
                (game, model_path, N_MCTS, tau_turns) for _ in range(PROCESSES)
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
            model.save(os.path.join("best-networks", f"model_{game.name}.h5"))
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

            print(
                f"Epoch: {epoch}/{final_epoch} | "
                f"Time: {total_time_r[0]}:{total_time_r[1]:02d}:{total_time_r[2]:02d} | "
                f'Loss: {dict_loss["Loss (Total)"][-1]:.4f} (Policy: {dict_loss["Loss (Policy)"][-1]:.4f}, '
                f'Value: {dict_loss["Loss (Value)"][-1]:.4f}, L2: {dict_loss["Loss (L2)"][-1]:.4f}) | '
                f"Replay buffer size: {len(replay_buffer)}"
            )

            learning_curve(6, 4)
            dashboard(16, 4)
            dashboard(16, 4, dashboard_epochs)

    if mode is "play":
        model = tf.keras.models.load_model(
            os.path.join("best-networks", f"model_{game.name}.h5")
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
                _, action, _, _ = drl.mcts(game, model, n_mcts_play, verbose=True)

            game.update(action)

            print(game.state)
            print("--------------")

        print(f"Result: {game.result}")

    if mode is "tournament":

        tournament_args = []
        for epoch_1 in tournament_range:
            for epoch_2 in tournament_range:
                tournament_args.append(
                    (game, epoch_1, epoch_2, n_mcts_play, len(tournament_args))
                )

        print(
            f"Game: {game.name} | Tournament games: {len(tournament_args)} | n_mcts_play: {n_mcts_play}"
        )

        with Pool(PROCESSES) as pool:
            pool_results = pool.starmap_async(tournament, tournament_args)
            tournament_logs = pool_results.get()

        print(
            f"Game: {game.name} | Tournament games: {len(tournament_args)} | n_mcts_play: {n_mcts_play}"
        )
        tournament_results = {
            i: {"Win": 0, "Draw": 0, "Lose": 0} for i in tournament_range
        }
        for p1_epoch, p2_epoch, game_result in tournament_logs:
            if game_result == 1:
                tournament_results[p1_epoch]["Win"] += 1
            elif game_result == 2:
                tournament_results[p1_epoch]["Lose"] += 1
            else:
                tournament_results[p1_epoch]["Draw"] += 1

            print(
                f"Player 1: {p1_epoch:3d} | Player 2: {p2_epoch:3d} | Result: {game_result}"
            )

        print("-------------------------")
        print(
            f"Game: {game.name} | Tournament games: {len(tournament_args)} | n_mcts_play: {n_mcts_play}"
        )
        for epoch in tournament_range:
            print(
                f"Epoch {epoch:3d} | Win: {tournament_results[epoch]['Win']} | "
                f"Draw: {tournament_results[epoch]['Draw']} | Lose: {tournament_results[epoch]['Lose']}"
            )
