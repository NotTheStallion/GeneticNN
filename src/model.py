from torch import nn


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


if __name__ == "__main__":
    chromosome = [2, 4, 4, 1]
    model = GeneticNN(chromosome)
    print(model.layers)
    
    chromosome = [90, 156, 200, 100, 50, 10, 1]
    model = GeneticNN(chromosome)
    print(model.layers)