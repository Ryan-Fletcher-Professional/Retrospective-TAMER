import argparse
import json
import math
import os
import torch
from environment.GLOBALS import *
from networks.Network import Net
import numpy as np
import matplotlib.pyplot as plt


def test_performance(net, env_name, plays):
    return -1


def test_frequency(log, is_gamma):
    bad = 0
    good = 0
    for press in log["feedback"]:
        if is_gamma:
            if press[0][0] > 0:
                bad += 1
            else:
                good += 1
        else:
            if press < 1:
                bad += 1
            else:
                good += 1
    return {"overall": (bad + good) / len(log["state_actions"]), "good_bad": (good / bad) if (bad > 0) else math.inf}


def test_timing(log, is_gamma):
    bads = []
    goods = []
    feedbacks = log["feedback"]
    for index in log["feedback_inds"]:
        true_index = len(feedbacks) + index
        if is_gamma:
            if feedbacks[true_index][0][0] > 0:
                bads.append(true_index)
            else:
                goods.append(true_index)
        else:
            if feedbacks[true_index] < 1:
                bads.append(true_index)
            else:
                goods.append(true_index)
    return {"overall": np.mean(np.array(bads + goods)), "good": np.mean(np.array(goods)), "bad": np.mean(np.array(bads)), "good_indeces": goods, "bad_indeces": bads}


def compare(test1, test2):
    return -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--load', default=None, type=str, help="directory of saved data to load")
    parser.add_argument('--save', default=None, type=str, help="directory in which to save the evaluations")
    parser.add_argument('--mode', default="frequency", type=str, help="performance/frequency/timing/compare")
    parser.add_argument('--display', default="False", type=str, help="whether to display already-processed results of the given mode")
    parser.add_argument('--mc_plays', default=1, type=int, help="number of runs for mountaincar performance evaluation")
    parser.add_argument('--ttt_plays', default=1, type=int, help="number of runs for tictactoe performance evaluation")
    parser.add_argument('--snake_plays', default=1, type=int, help="number of runs for snake performance evaluation")

    args = parser.parse_args()

    # TODO :
    #   Implement the test methods

    mc_process = []
    ttt_process = []
    snake_process = []
    dumps = {"mc": mc_process, "ttt": ttt_process, "snake": snake_process}
    compared = []
    to_display = {}

    for filename in os.listdir(args.load):
        filepath = args.load + "/" + filename
        name, extension = os.path.splitext(filename)
        if name is None:
            print("SKIPPED (no name):", filename)
            continue

        if args.display == "True":
            if (args.mode == "performance") and ("_performance" in name):
                with open(filepath, 'r') as file:
                    performances = json.load(file)
                    for performance in performances:
                        run_name = performance[0].replace("_1", "").replace("_2", "").replace("_3", "")
                        episode_index = 0 if ("_1" in performance[0]) else (1 if ("_2" in performance[0]) else 2)
                        if run_name not in to_display.keys():
                            to_display[run_name] = [None, None, None]
                        to_display[run_name][episode_index] = (performance[1]["mean"], performance[1]["std"])
            elif (args.mode == "frequency") and ("_frequency" in name):
                with open(filepath, 'r') as file:
                    frequencies = json.load(file)
                    for frequency in frequencies:
                        run_name = frequency[0].replace("_1", "").replace("_2", "").replace("_3", "")
                        episode_index = 0 if ("_1" in frequency[0]) else (1 if ("_2" in frequency[0]) else 2)
                        if run_name not in to_display.keys():
                            to_display[run_name] = [None, None, None]
                        to_display[run_name][episode_index] = (frequency[1]["overall"], frequency[1]["good_bad"])
            elif (args.mode == "timing") and ("_timing" in name):
                with open(filepath, 'r') as file:
                    timings = json.load(file)
                    for timing in timings:
                        run_name = timing[0].replace("_1", "").replace("_2", "").replace("_3", "")
                        episode_index = 0 if ("_1" in timing[0]) else (1 if ("_2" in timing[0]) else 2)
                        if run_name not in to_display.keys():
                            to_display[run_name] = [None, None, None]
                        to_display[run_name][episode_index] = (timing[1]["overall"], timing[1]["good"], timing[1]["bad"])
            continue

        env_name = MOUNTAIN_CAR_MODE if ("_mc_" in name) else (TTT_MODE if ("_ttt_" in name) else (SNAKE_MODE if ("_snake_" in name) else None))
        if env_name == MOUNTAIN_CAR_MODE:
            to_append = mc_process
        elif env_name == TTT_MODE:
            if (args.mode == "frequency") or (args.mode == "timing"):
                print("Skipped (TTT):", filename)
                continue
            to_append = ttt_process
        elif env_name == SNAKE_MODE:
            to_append = snake_process
        else:
            print("ERROR! Unrecognized env_name for:", filename)
            continue

        if args.mode == "compare":
            if filename in compared:
                continue
            if extension != ".json":
                print("SKIPPED (filetype):", filename)
                continue
            if "_live" not in filename:
                if "_retrospective" not in filename:
                    print("SKIPPED (not live/retro):", filename)
                continue

            for filename2 in os.listdir(args.load):
                filepath2 = args.load + "/" + filename2
                if not (("_retrospective" in filename2) and
                        (filename.replace("_live", "") == filename2.replace("_retrospective", ""))):
                    continue

                compared.append(filename)
                compared.append(filename2)

                with open(filepath, 'r') as file:
                    with open(filepath2, 'r') as file2:
                        to_append.append((name.replace("_live", ""), compare(json.load(file), json.load(file2))))
        else:
            if (args.mode == "performance") and ((extension == ".param") or (extension == ".pt")):
                net = Net(env_name)
                net.load_state_dict(torch.load(filename))
                to_append.append((name, test_performance(net, env_name, args.mc_plays if (env_name == MOUNTAIN_CAR_MODE) else (args.ttt_plays if (env_name == TTT_MODE) else args.snake_plays))))
            elif ((args.mode == "frequency") or (args.mode == "timing")) and (extension == ".json"):
                with open(filepath, 'r') as file:
                    #print("SAVING:", name)
                    to_append.append((name, test_frequency(json.load(file), env_name != TTT_MODE) if (args.mode == "frequency") else test_timing(json.load(file), env_name != TTT_MODE)))
            else:
                print("SKIPPED (filetype):", filename)

    if args.display == "True":
        if args.mode == "performance":
            for key, value_list in to_display.items():
                means, stdevs = zip(*value_list)
                x_positions = np.arange(len(means))
                plt.errorbar(x_positions, means, yerr=stdevs, label=key, capsize=5, marker='o')
            plt.xticks(ticks=np.arange(3), labels=['Checkpoint 1', 'Checkpoint 2', 'Checkpoint 3'])
            plt.xlabel('Checkpoint')
            plt.ylabel('Mean Performance')
            plt.title('Performance of Trained Networks')
            plt.legend()
            plt.grid(True)
            plt.show()
        elif args.mode == "frequency":
            print(to_display)
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

            for key, value_list in to_display.items():
                values = [t[0] for t in value_list]
                x_positions = np.arange(len(values))
                axes[0].plot(x_positions, values, marker='o', label=key)

            for key, value_list in to_display.items():
                values_unfixed = [t[1] for t in value_list]
                values = [value if (value < math.inf) else max(values_unfixed) for value in values_unfixed]
                x_positions = np.arange(len(values))
                axes[1].plot(x_positions, values, marker='o', label=key)

            axes[0].set_xticks(np.arange(3))
            axes[0].set_xticklabels(['Checkpoint 1', 'Checkpoint 2', 'Checkpoint 3'])
            axes[0].set_xlabel('Checkpoint')
            axes[0].set_ylabel('Overall Feedback Frequency')
            axes[0].set_title('Overall Feedback Frequency for Each Episode')
            axes[0].legend()
            axes[0].grid(True)

            axes[1].set_xticks(np.arange(3))
            axes[1].set_xticklabels(['Checkpoint 1', 'Checkpoint 2', 'Checkpoint 3'])
            axes[1].set_xlabel('Checkpoint')
            axes[1].set_ylabel('Good:Bad Feedback Frequency')
            axes[1].set_title('Good/Bad Feedback Frequency for Each Episode')
            axes[1].legend()
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()
        elif args.mode == "timing":
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))

            for key, value_list in to_display.items():
                values = [t[0] for t in value_list]
                x_positions = np.arange(len(values))
                axes[0].plot(x_positions, values, marker='o', label=key)

            for key, value_list in to_display.items():
                values = [t[1] for t in value_list]
                x_positions = np.arange(len(values))
                axes[1].plot(x_positions, values, marker='o', label=key)

            for key, value_list in to_display.items():
                values = [t[2] for t in value_list]
                x_positions = np.arange(len(values))
                axes[2].plot(x_positions, values, marker='o', label=key)

            axes[0].set_xticks(np.arange(3))
            axes[0].set_xticklabels(['Checkpoint 1', 'Checkpoint 2', 'Checkpoint 3'])
            axes[0].set_xlabel('Checkpoint')
            axes[0].set_ylabel('Overall Feedback Timing')
            axes[0].set_title('Overall Feedback Timing for Each Episode')
            axes[0].legend()
            axes[0].grid(True)

            axes[1].set_xticks(np.arange(3))
            axes[1].set_xticklabels(['Checkpoint 1', 'Checkpoint 2', 'Checkpoint 3'])
            axes[1].set_xlabel('Checkpoint')
            axes[1].set_ylabel('Good Feedback Timing')
            axes[1].set_title('Good Feedback Timing for Each Episode')
            axes[1].legend()
            axes[1].grid(True)

            axes[2].set_xticks(np.arange(3))
            axes[2].set_xticklabels(['Checkpoint 1', 'Checkpoint 2', 'Checkpoint 3'])
            axes[2].set_xlabel('Checkpoint')
            axes[2].set_ylabel('Bad Feedback Timing')
            axes[2].set_title('Bad Feedback Timing for Each Episode')
            axes[2].legend()
            axes[2].grid(True)

            plt.tight_layout()
            plt.show()
        else:
            print("ERROR! Unrecognized mode:", args.mode)
    else:
        for env_name in dumps.keys():
            with open(args.save + "/" + env_name + "_" + args.mode + ".json", 'w') as file:
                json.dump(dumps[env_name], file)
