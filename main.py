import argparse
import torch
from environment.GLOBALS import *
from networks.Network import Net
from training.offline import offline_wrapper
from training.online import collect_live_data


device = torch.device('cpu')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_plays', default=5, type=int, help="number of play runs to collect")
    parser.add_argument('--mode', default=DEFAULT_MODE, type=str, help="Which game to do")
    parser.add_argument('--snake_max_fps', default=10, type=int, help="max fps at which snake game should run")
    parser.add_argument('--frame_limit', default=200, type=int, help="number of frames before episode cuts off")
    #parser.add_argument('--num_evals', default=6, type=int, help="number of times to run policy after training for evaluation")
    args = parser.parse_args()

    net = Net(args.mode)
    for i in range(args.num_plays):
        data = collect_live_data(net, env_name=args.mode, frame_limit=args.frame_limit, snake_max_fps=args.snake_max_fps)
        print("Data for run " + str(i + 1) + ":\n" + str(data))

    #data=offline_wrapper(net, env_name=args.mode, frame_limit=args.frame_limit, snake_max_fps=args.snake_max_fps)
    #print(data)
    # collect human
