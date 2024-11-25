import torch
import numpy as np


import torch.nn as nn
import torch.optim as optim

class Network(nn.Module):
    def __init__(self, input_shape, classes, DNA_param, epochs):
        super(Network, self).__init__()
        
        self.architecture_DNA = []  # to save current parameters
        self.fitness = []
        self.acc_history = []
        self.input_shape = input_shape
        self.classes = classes
        self.epochs = epochs

        # unfold DNA_parameters:
        depth = DNA_param[0]
        neurons_per_layer = DNA_param[1]
        activations = DNA_param[2]
        optimizers = DNA_param[3]
        losses = DNA_param[4]

        layers = []
        network_depth = np.random.choice(depth)
        self.architecture_DNA.append(network_depth)

        for i in range(network_depth):
            if i == 0:
                neurons = np.random.choice(neurons_per_layer)
                activation = np.random.choice(activations)
                self.architecture_DNA.append([neurons, activation])
                layers.append(nn.Linear(self.input_shape, neurons))
                layers.append(self.get_activation(activation))
            elif i == network_depth - 1:
                activation = np.random.choice(activations)
                self.architecture_DNA.append(activation)
                layers.append(nn.Linear(neurons, self.classes))
                layers.append(self.get_activation(activation))
            else:
                neurons = np.random.choice(neurons_per_layer)
                activation = np.random.choice(activations)
                self.architecture_DNA.append([neurons, activation])
                layers.append(nn.Linear(neurons, neurons))
                layers.append(self.get_activation(activation))

        self.model = nn.Sequential(*layers)

        loss = np.random.choice(losses)
        optimizer = np.random.choice(optimizers)
        self.architecture_DNA.append([loss, optimizer])
        self.loss_fn = self.get_loss(loss)
        self.optimizer = self.get_optimizer(optimizer, self.model.parameters())

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def get_loss(self, loss):
        if loss == 'cross_entropy':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss}")

    def get_optimizer(self, optimizer, parameters):
        if optimizer == 'adam':
            return optim.Adam(parameters)
        elif optimizer == 'sgd':
            return optim.SGD(parameters)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def create_children(self, children_DNA):
        layers = []
        children_depth = children_DNA[0]

        for i in range(children_depth):
            if i == 0:
                layers.append(nn.Linear(self.input_shape, children_DNA[1][0]))
                layers.append(self.get_activation(children_DNA[1][1]))
            elif i == children_depth - 1:
                layers.append(nn.Linear(children_DNA[i][0], self.classes))
                layers.append(self.get_activation(children_DNA[children_depth]))
            else:
                layers.append(nn.Linear(children_DNA[i][0], children_DNA[i+1][0]))
                layers.append(self.get_activation(children_DNA[i+1][1]))

        self.model = nn.Sequential(*layers)
        self.loss_fn = self.get_loss(children_DNA[-1][0])
        self.optimizer = self.get_optimizer(children_DNA[-1][1], self.model.parameters())
        self.architecture_DNA = children_DNA

    def give_fitness(self):
        return self.fitness

    def train(self, X_train, y_train):
        self.model.train()
        for epoch in range(self.epochs):
            for inputs, labels in zip(X_train, y_train):
                inputs = torch.tensor(inputs, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.long)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def test(self, X_test, y_test):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in zip(X_test, y_test):
                inputs = torch.tensor(inputs, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.long)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.fitness = correct / total
        self.acc_history.append(self.fitness)

    def give_DNA(self):
        return self.architecture_DNA

    def architecture(self):
        print(self.model)