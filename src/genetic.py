import torch

DNA_parameter = [[5,6,7,8,9,10],
                 [16,32,64,128,256,512,1024],
                 ["tanh","softmax","relu","sigmoid","elu","selu","softplus","softsign","hard_sigmoid","linear"], #"leakyrelu",
                 ["sgd","rmsprop","adagrad","adadelta","adam","adamax","nadam"],
                 ["mean_squared_error","mean_absolute_error","mean_absolute_percentage_error","mean_squared_logarithmic_error","squared_hinge","hinge","categorical_hinge","logcosh","categorical_crossentropy","binary_crossentropy","kullback_leibler_divergence","poisson","cosine_proximity"] #"sparse_categorical_crossentropy",
                ]

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, generations = 50, Epochs = 2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.training_epochs = Epochs
        self.population = None
        self.children_population_DNA = []
        self.acces = []
        self.norm_acces = []
        
    def create_population(self):
        self.population = [Network(image_vector_size, num_classes, DNA_parameter,self.training_epochs) for i in range(self.population_size)]
    
    def train_generation(self):
        for member in self.population:
                member.train()
                
    def predict(self):
        for member in self.population:
                member.test()
                self.acc.append(member.give_fitness())
    
    def normalize(self):
        sum_ = sum(self.acc)
        self.norm_acc = [i/sum_ for i in self.acc] 
        #print("\nNormalization sum: ",sum(self.norm_acc))
        #assert sum(self.norm_acc) == 1
        
    def clear_losses(self):
        self.norm_acc = []
        self.acc = []
        
    def mutate(self):
        for child_DNA in self.children_population_DNA:
            for i in range(len(child_DNA)):
                if np.random.random() < self.mutation_rate:
                    print("\nMutation!")
                    if i == 0:
                        new_depth = np.random.choice(DNA_parameter[0])
                        child_DNA[0] = new_depth
                    
                    if i == len(child_DNA)-2:
                        new_output_activation = np.random.choice(DNA_parameter[2])
                        child_DNA[-2] = new_output_activation
                    
                    if i == len(child_DNA)-1:
                        # random flip if loss or activation shall be changed
                        if np.random.random() < 0.5:
                            new_loss = np.random.choice(DNA_parameter[4])
                            child_DNA[-1][0] = new_loss
                        else:
                            new_optimizer = np.random.choice(DNA_parameter[3])
                            child_DNA[-1][1] = new_optimizer
                    if i != 0 and i !=len(child_DNA)-2 and i != len(child_DNA)-1:
                    #else:
                        # 3/2 flif if number of neurons or activation function mutates:
                        #print(child_DNA)
                        if np.random.random() < 0.33:
                            #print(child_DNA[i][1])
                            new_activation = np.random.choice(DNA_parameter[2])
                            #print(new_activation)
                            child_DNA[i][1] = new_activation
                        else:
                            #print(child_DNA[i][0])
                            new_neuron_count = np.random.choice(DNA_parameter[1])
                            child_DNA[i][0] = new_neuron_count
                            #print(new_neuron_count)
                    #print("After mutation ", child_DNA)

    def reproduction(self):
        """ 
        Reproduction through midpoint crossover method 
        """
        population_idx = [i for i in range(len(self.population))]
        for i in range(len(self.population)):
        #selects two parents probabilistic accroding to the fitness score
            if sum(self.norm_acc) != 0:
                parent1 = np.random.choice(population_idx, p = self.norm_acc)
                parent2 = np.random.choice(population_idx, p = self.norm_acc)
            else:
              # if there are no "best" parents choose randomly 
                parent1 = np.random.choice(population_idx)
                parent2 = np.random.choice(population_idx)

            # picking random midpoint for crossing over name/DNA
            parent1_DNA = self.population[parent1].give_DNA()
            parent2_DNA = self.population[parent2].give_DNA()
            #print(parent1_DNA)
            
            mid_point_1 = np.random.choice([i for i in range(2,len(parent1_DNA)-2)])
            mid_point_2 = np.random.choice([i for i in range(2,len(parent2_DNA)-2)])
            # adding DNA-Sequences of the parents to final DNA
            child_DNA = parent1_DNA[:mid_point_1] + parent2_DNA[mid_point_2:]
            new_nn_depth = len(child_DNA)-2 # minus 2 because of depth parameter[0] and loss parameter[-1]
            child_DNA[0] = new_nn_depth
            self.children_population_DNA.append(child_DNA)
        # old population gets the new and proper weights
        self.mutate()
        keras.backend.clear_session() ## delete old models to free memory
        for i in range(len(self.population)):
            self.population[i].create_children(self.children_population_DNA[i])
        
        
    
    def run_evolution(self):
        for episode in range(self.generations):
            print("\n--- Generation {} ---".format(episode))
            self.clear_losses()
            self.train_generation()
            self.predict()
            if episode != self.generations -1:
                self.normalize()
                self.reproduction()
                
            else:
                pass
            self.children_population_DNA = []
        # plotting history:
        for a in range(self.generations):
            for member in self.population:
                plt.plot(member.acc_history)
        plt.xlabel("Generations")
        plt.ylabel("Accuracy")
        plt.show()