import torch
import numpy as np
from environment.GLOBALS import *
from scipy.stats import gamma



def make_state_action(state, action, mode):
    '''
    Takes in the state as obs np array
    and the action as an integer
    returns np array of state and one-hot encoded action
    '''
    # one hot encode the actions
    n_actions = ACTION_SIZES[mode]
    one_hot_action = np.zeros(n_actions)
    one_hot_action[action] = 1
    # append the state, actions to obs_data
    state_action = np.append(state, one_hot_action)

    return state_action

def make_training_data_with_gamma(state_action_data, feedback, time_data, feedback_delta):
    '''
    Inputs:
    state_action_data - every single state_action pair from start of game
                        to until feedback was given
    feedback - binary feedback given
    time_data - timestamps (in seconds) for every (state, action)
    feedback_delta - timestamp of the feedback recorded

    Outputs:
    input tensor and output tensor that contains feedback credits assigned based
    on a gamma function, to account for human reaction time.
    '''
    alpha = MOUNTAIN_CAR_GAMMA['alpha']
    loc = MOUNTAIN_CAR_GAMMA['loc']
    scale = MOUNTAIN_CAR_GAMMA['scale']
    state_action_data = np.asarray(state_action_data)

    # copying time_data so I don't modify the original that keeps track
    # of time for each action
    # reverse the time so feedback happens at time 0, and each observation
    # corresponds to how much time has passed from that observation until
    # feedback was recorded
    time_data_mod = np.copy(time_data)
    time_data_mod = np.append(time_data_mod, feedback_delta)
    time_data_mod = time_data_mod[-1]-time_data_mod
    n = len(time_data_mod)

    # use gamma function to decide on how feedback credits should be assigned
    credits = gamma.cdf(time_data_mod[0: n-1], alpha, loc, scale) \
            - gamma.cdf(time_data_mod[1: n], alpha, loc, scale)
    #print("-------------------credits before cutoff", credits)
    # for an almost 0 credit, better to not pass it to network at all
    # find the index of that credit cutoff
    credits_cutoff_ind = np.argmax(credits>GAMMA_CREDIT_CUTOFF)
    credits = credits[credits_cutoff_ind:]

    # this scales the credits so the max is 1
    if max(credits)!=0:
        credits/= max(credits)
    #print("-------------------credits after cutoff and scaling", credits)
    # use the credit cutoff for the network input data as well
    state_action_data = state_action_data[credits_cutoff_ind:]

    # also cut off the right tail where everything is 0 because its under
    # our estimated minimum human reaction time
    if 0 in credits:
        credits_cutoff_ind = np.argmax(credits==0)
        credits = credits[:credits_cutoff_ind]
        state_action_data = state_action_data[:credits_cutoff_ind]

    # convert this where each credit is [0, credit] or [credit, 0] depending
    # on feedback
    network_output = np.zeros((len(credits), 2))
    network_output[:, feedback] = credits
    # print("Credits:", network_output)

    input_tensor = torch.as_tensor(state_action_data, dtype=torch.float32).reshape(state_action_data.shape[0], -1)
    output_tensor = torch.as_tensor(network_output).reshape(network_output.shape[0], -1)

    # print("IN:", input_tensor)
    # print("OUT:", output_tensor)

    return input_tensor, output_tensor


def make_training_data(input, output):
    '''
    Takes in a single state,action pair and one binary feedback.
    Outputs the input and output formatted as appropriatly shaped
    tensors.
    '''
    input_tensor = torch.as_tensor(input, dtype=torch.float32)
    output_one_hot = np.zeros((1, 2))
    output_one_hot[0, output] = 1
    output_tensor = torch.as_tensor(output_one_hot)

    return(torch.unsqueeze(input_tensor, 0), output_tensor)
