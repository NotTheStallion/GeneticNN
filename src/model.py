from torch import nn, optim
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


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
    
    def train_model(self, train_loader, val_loader, loss_fn, optimizer, epochs, patience=5):
        self.train_losses, self.validation_losses = [], []
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(epochs):
            self.train()
            epoch_train_loss = 0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self(x_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            
            self.train_losses.append(epoch_train_loss / len(train_loader))
            val_loss = self.validate_model(val_loader, loss_fn)
            self.validation_losses.append(val_loss)
            
            # print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return self.train_losses, self.validation_losses
    
    def validate_model(self, val_loader, loss_fn):
        self.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                y_pred = self(x_batch)
                loss = loss_fn(y_pred, y_batch)
                val_loss += loss.item()
        return val_loss / len(val_loader)
    
    def test_model(self, test_loader, loss_fn):
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                y_pred = self(x_batch)
                loss = loss_fn(y_pred, y_batch)
                test_loss += loss.item()
        return test_loss / len(test_loader)
    
    ## Genetic Algorithm methods
    
    @staticmethod
    def random_chromosome(input_size, output_size, max_layers=10, max_neurons_power=7):
        """Generate a random chromosome with neurons as powers of 2."""
        num_layers = np.random.randint(2, max_layers + 1)
        chromosome = [input_size]
        chromosome.extend(2 ** np.random.randint(1, max_neurons_power + 1) for _ in range(num_layers - 2))
        chromosome.append(output_size)
        return chromosome
    
    @staticmethod
    def mutate_chromosome(chromosome, mutation_rate=0.1, max_neurons_power=7):
        """Apply mutation to a chromosome with neurons as powers of 2."""
        for i in range(1, len(chromosome) - 1):  # Do not mutate input/output sizes
            if np.random.rand() < mutation_rate:
                chromosome[i] = 2 ** np.random.randint(1, max_neurons_power + 1)
        if len(chromosome) > 3 and np.random.rand() < mutation_rate:
            del chromosome[np.random.randint(1, len(chromosome) - 1)]
        return chromosome
    
    @staticmethod
    def crossover(parent1, parent2):
        """Perform single-point crossover."""
        point = np.random.randint(1, min(len(parent1), len(parent2)))
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def evaluate_fitness(self, train_loader, val_loader, loss_fn, optimizer, epochs, patience):
        """Evaluate fitness of a chromosome."""
        train_losses, val_losses = self.train_model(train_loader, val_loader, loss_fn, optimizer, epochs, patience)
        return (-self.validation_losses[-1])-(len(self.chromosome)/10)-(np.mean(self.chromosome)/2**7), train_losses, val_losses  # Use the final validation loss as the fitness score
    
    @staticmethod
    def select_parents(population, fitness_scores, num_parents):
        """Select parents based on fitness scores (lower is better)."""
        sorted_indices = np.argsort(fitness_scores)
        return [population[i] for i in sorted_indices[:num_parents]]





if __name__ == "__main__":
    # Example configuration
    chromosome = [2, 4, 4, 1]
    model = GeneticNN(chromosome)
    
    # Dummy data
    x_train = torch.tensor(np.random.rand(100, 2), dtype=torch.float32)
    y_train = torch.tensor(np.random.rand(100, 1), dtype=torch.float32)
    x_val = torch.tensor(np.random.rand(20, 2), dtype=torch.float32)
    y_val = torch.tensor(np.random.rand(20, 1), dtype=torch.float32)
    
    # Helper function to create dataloaders
    def create_dataloaders(x_train, y_train, x_val, y_val, batch_size=32):
        train_data = TensorDataset(x_train, y_train)
        val_data = TensorDataset(x_val, y_val)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(x_train, y_train, x_val, y_val)
    
    # Training configuration
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 50
    
    # Train the model
    train_losses, val_losses = model.train_model(train_loader, val_loader, loss_fn, optimizer, epochs, patience=5)
