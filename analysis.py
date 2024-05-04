import argparse
import json
import os
import torch
from environment.GLOBALS import *
from networks.Network import Net


def test_performance(net, env_name, plays):
    return -1


def test_frequency(log):
    return -1


def test_timing(log):
    return -1


def compare(test1, test2):
    return -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--load', default=None, type=str, help="directory of saved data to load")
    parser.add_argument('--save', default=None, type=str, help="directory in which to save the evaluations")
    parser.add_argument('--mode', default="frequency", type=str, help="performance/frequency/timing/compare")
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

    for filename in os.listdir(args.load):
        name, extension = os.path.splitext(filename)
        if name is None:
            print("SKIPPED (no name):", filename)
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
                if not (("_retrospective" in filename2) and
                        (filename.replace("_live", "") == filename2.replace("_retrospective", ""))):
                    continue

                compared.append(filename)
                compared.append(filename2)

                with open(filename, 'r') as file:
                    with open(filename2, 'r') as file2:
                        to_append.append((name.replace("_live", ""), compare(json.load(file), json.load(file2))))
        else:
            if (args.mode == "performance") and ((extension == ".param") or (extension == ".pt")):
                net = Net(env_name)
                net.load_state_dict(torch.load(filename))
                to_append.append((name, test_performance(net, env_name, args.mc_plays if (env_name == MOUNTAIN_CAR_MODE) else (args.ttt_plays if (env_name == TTT_MODE) else args.snake_plays))))
            elif extension == ".json":
                with open(filename, 'r') as file:
                    to_append.append((name, test_frequency(json.load(file)) if (args.mode == "frequency") else test_timing(json.load(file))))
            else:
                print("SKIPPED (filetype):", filename)

    for env_name in dumps.keys():
        with open(args.save + "/" + env_name + "_" + args.mode + ".json", 'w') as file:
            json.dump(dumps[env_name], file)
