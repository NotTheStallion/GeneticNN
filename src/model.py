from torch import nn
import torch


class GeneticNN(nn.Module):
    def __init__(self, chromosome):
        super(GeneticNN, self).__init__()
        self.chromosome = chromosome
        self.layers = nn.ModuleList()
        
        for idx_lyr in range(len(chromosome) - 1):
            self.layers.append(nn.Linear(chromosome[idx_lyr], chromosome[idx_lyr + 1]))
            
            if idx_lyr < len(chromosome) - 2:
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Sigmoid())
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def train_model(self, x_train, y_train, x_test, y_test, loss_fn, optimizer, epochs):
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self(x_train)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()
            
            test_losses.append(self.test_model(x_test, y_test, loss_fn))
            train_losses.append(loss.item())
            # print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
        return train_losses, test_losses
    
    def test_model(self, x, y, criterion):
        with torch.no_grad():
            y_pred = self(x)
            loss = criterion(y_pred, y)
            # print(f"Test Loss: {loss.item()}")
        return loss.item()


if __name__ == "__main__":
    chromosome = [2, 4, 4, 1]
    model = GeneticNN(chromosome)
    print(model.layers)
    
    chromosome = [90, 156, 200, 100, 50, 10, 1]
    model = GeneticNN(chromosome)
    print(model.layers)