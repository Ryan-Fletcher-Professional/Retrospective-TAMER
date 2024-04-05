import gymnasium as gym
import argparse
import pygame
from teleop import collect_demos
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from GLOBALS import *

####################################################################################
# Much of this code was adapted from Scott Niekum's COMPSCI 690S assignments 1 & 3 #
####################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# TODO : Use for torchifying trajectories?
def torchify(sas_pairs):
    # states = []
    # actions = []
    # next_states = []
    # for s, a, s2 in sas_pairs:
    #     states.append(s)
    #     actions.append(a)
    #     next_states.append(s2)
    #
    # states = np.array(states)
    # actions = np.array(actions)
    # next_states = np.array(next_states)
    #
    # obs_torch = torch.from_numpy(np.array(states)).float().to(device)
    # obs2_torch = torch.from_numpy(np.array(next_states)).float().to(device)
    # acs_torch = torch.from_numpy(np.array(actions)).long().to(device)
    #
    # return obs_torch, acs_torch, obs2_torch
    pass


def tamer():
    # TODO
    # wrapper for train, among other stuff
    pass


# function to train a vanilla policy gradient agent. By default designed to work with Cartpole
def train(env_name=DEFAULT_MODE, hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, render=False, reward=None, checkpoint=False, checkpoint_dir="\."):
    # # make environment, check spaces, get obs / act dims
    # env = gym.make(env_name)
    # assert isinstance(env.observation_space, Box), \
    #     "This example only works for envs with continuous state spaces."
    # assert isinstance(env.action_space, Discrete), \
    #     "This example only works for envs with discrete action spaces."
    #
    # obs_dim = env.observation_space.shape[0]
    # n_acts = env.action_space.n
    #
    # # make core of policy network
    # logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])
    #
    # # make function to compute action distribution
    # def get_policy(obs):
    #     logits = logits_net(obs)
    #     return Categorical(logits=logits)
    #
    # # make action selection function (outputs int actions, sampled from policy)
    # def get_action(obs):
    #     return get_policy(obs).sample().item()
    #
    # # make loss function whose gradient, for the right data, is policy gradient
    # def compute_loss(obs, act, weights):
    #     logp = get_policy(obs).log_prob(act)
    #     return -(logp * weights).mean()
    #
    # # make optimizer
    # optimizer = Adam(logits_net.parameters(), lr=lr)
    #
    # # for training policy
    # def train_one_epoch():
    #     # make some empty lists for logging.
    #     batch_obs = []  # for observations
    #     batch_acts = []  # for actions
    #     batch_weights = []  # for reward-to-go weighting in policy gradient
    #     batch_rets = []  # for measuring episode returns
    #     batch_lens = []  # for measuring episode lengths
    #
    #     # reset episode-specific variables
    #     obs, _ = env.reset()  # first obs comes from starting distribution
    #     done = False  # signal from environment that episode is over
    #     ep_rews = []  # list for rewards accrued throughout ep
    #
    #     # render first episode of each epoch
    #     finished_rendering_this_epoch = False
    #
    #     # collect experience by acting in the environment with current policy
    #     while True:
    #
    #         # rendering
    #         if (not finished_rendering_this_epoch) and render:
    #             env.render()
    #
    #         # save obs
    #         batch_obs.append(obs.copy())
    #
    #         # act in the environment
    #         act = get_action(torch.as_tensor(obs, dtype=torch.float32))
    #         obs, rew, terminated, truncated, _ = env.step(act)
    #         done = terminated or truncated
    #
    #         if reward is not None:
    #             # replace reward with predicted reward from neural net
    #             device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #             torchified_state = torch.from_numpy(obs).float().to(device)
    #             r = reward.predict_return(torchified_state.unsqueeze(0)).item()
    #             rew = r
    #
    #         # save action, reward
    #         batch_acts.append(act)
    #         ep_rews.append(rew)
    #
    #         if done:
    #             # if episode is over, record info about episode
    #             ep_ret, ep_len = sum(ep_rews), len(ep_rews)
    #             batch_rets.append(ep_ret)
    #             batch_lens.append(ep_len)
    #
    #             # the weight for each logprob(a_t|s_t) is reward-to-go from t
    #             batch_weights += list(reward_to_go(ep_rews))
    #
    #             # reset episode-specific variables
    #             obs, _ = env.reset()
    #             done = False
    #             ep_rews = []
    #
    #             # won't render again this epoch
    #             finished_rendering_this_epoch = True
    #
    #             # end experience loop if we have enough of it
    #             if len(batch_obs) > batch_size:
    #                 break
    #
    #     # take a single policy gradient update step
    #     optimizer.zero_grad()
    #     batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
    #                               act=torch.as_tensor(batch_acts, dtype=torch.int32),
    #                               weights=torch.as_tensor(batch_weights, dtype=torch.float32)
    #                               )
    #     batch_loss.backward()
    #     optimizer.step()
    #     return batch_loss, batch_rets, batch_lens
    #
    # # training loop
    # for i in range(epochs):
    #     batch_loss, batch_rets, batch_lens = train_one_epoch()
    #     if reward is not None:
    #         print('epoch: %3d \t loss: %.3f \t predicted return: %.3f \t ep_len (gt reward): %.3f' %
    #               (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
    #     else:
    #         print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
    #               (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
    #
    #     if checkpoint:
    #         # checkpoint after each epoch
    #         print("!!!!!! checkpointing policy !!!!!!")
    #         torch.save(logits_net.state_dict(), checkpoint_dir + '/policy_checkpoint' + str(i) + '.params')
    #
    # # always at least checkpoint at end of training
    # if not checkpoint:
    #     torch.save(logits_net.state_dict(), checkpoint_dir + '/final_policy.params')
    pass


# evaluate learned policy
def evaluate_policy(pi, num_evals, env_name=DEFAULT_MODE, human_render=True):
    # if human_render:
    #     env = gym.make(env_name, render_mode='human')
    # else:
    #     env = gym.make(env_name)
    #
    # policy_returns = []
    # for i in range(num_evals):
    #     done = False
    #     total_reward = 0
    #     obs, _ = env.reset()
    #     while not done:
    #         # take the action that the network assigns the highest logit value to
    #         # Note that first we convert from numpy to tensor and then we get the value of the
    #         # argmax using .item() and feed that into the environment
    #         action = torch.argmax(pi(torch.from_numpy(obs).unsqueeze(0))).item()
    #         # print(action)
    #         obs, rew, terminated, truncated, info = env.step(action)
    #         done = terminated or truncated
    #         total_reward += rew
    #     print("reward for evaluation", i, total_reward)
    #     policy_returns.append(total_reward)
    #
    # print("average policy return", np.mean(policy_returns))
    # print("min policy return", np.min(policy_returns))
    # print("max policy return", np.max(policy_returns))
    pass


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('--num_demos', default=1, type=int, help="number of human demonstrations to collect")
    # parser.add_argument('--num_bc_iters', default=100, type=int, help="number of iterations to run BC")
    # parser.add_argument('--num_inv_dyn_iters', default=1000, type=int,
    #                     help="number of iterations to train inverse dynamics model")
    # parser.add_argument('--num_evals', default=6, type=int,
    #                     help="number of times to run policy after training for evaluation")
    #
    # args = parser.parse_args()
    #
    # # process demos
    # obs, ground_truth_acts, next_obs = torchify(demos)
    #
    # # TODO: ADD CODE TO TRAIN INVERSE DYNAMICS MODEL AND ESTIMATE ACTIONS
    # estimated_acts = inverse_dynamics(obs, next_obs, args.num_inv_dyn_iters)
    #
    # # train policy WITHOUT ground truth actions
    # pi = PolicyNetwork()
    # train_policy(obs, estimated_acts, pi, args.num_bc_iters)
    #
    # # evaluate learned policy
    # evaluate_policy(pi, args.num_evals)
    pass
