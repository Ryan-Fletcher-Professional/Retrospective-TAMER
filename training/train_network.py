from torch.optim import Adam
import torch.nn as nn


def train_network(net, input, output, lr=0.2):
    '''
    Takes correctly formatted input and output tensors and
    trains the given network
    '''
    # print(input.numpy(), output.numpy())

    optimizer = Adam(net.parameters(), lr=lr)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    #run action,state through policy to get predicted logits for classifying action
    pred_action_logits = net.predict(input)
    #now compute loss
    loss = loss_criterion(pred_action_logits, output)
    #back propagate the error through the network
    loss.backward()
    #perform update on policy parameters
    optimizer.step()
    #print("Updating network. Input: ", input.numpy(), "\nOutput: ", output.numpy())
