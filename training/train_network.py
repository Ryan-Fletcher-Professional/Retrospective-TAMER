from torch.optim import Adam
import torch.nn as nn
import torch


def train_network(net, input, output, lr=0.2):
    '''
    Takes correctly formatted input and output tensors and
    trains the given network
    '''
    # print("---------------------")
    # print(input.numpy(), output.numpy())
    # print("---------------------")
    optimizer = Adam(net.parameters(), lr=lr)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    #run action,state through policy to get predicted logits for classifying action
    prediction_logits = net.predict(input)
    # print("OG OUTPUT:", output)
    #output /= torch.sum(output)# output = nn.functional.softmax(output, dim=0)
    #now compute loss
    # print("PREDICTED LOGITS:", prediction_logits)
    # print("OUTPUT:", output)
    loss = loss_criterion(prediction_logits, output)
    #back propagate the error through the network
    loss.backward()
    #perform update on policy parameters
    optimizer.step()
    # for layer in net.parameters():
    #     print("GRAD:\n" + str(layer.grad))
    #print("Updating network. Input: ", input.numpy(), "\nOutput: ", output.numpy())

    # for layer in net.parameters():
    #     print(layer.data)
