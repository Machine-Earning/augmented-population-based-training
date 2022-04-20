############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Assginment: Term Paper
#   Date: 4/10/2022
#   file: apbt.py
#   Description: main class for the augmented 
#   population based training algorithm
#############################################################

# imports
from secrets import randbelow
from ann import ANN
import random

# Augmentated Population Based Training
class APBT:
    '''
    Augmentated Population Based Training
    Builts on top of Deepmind PBT algorithm
    and augments it to also optimize the 
    Neural network arcthiecture
    
    '''
    
    def __init__(self, k: int, 
        training: str, 
        testing: str, 
        attributes: str, 
        weights_path: str, 
        debug) -> None:
        '''
        Initialize the APBT class
        '''
        self.k = k
        self.population = [None for _ in range(k)]
        self.hyperparams = [None for _ in range(k)]
        self.perfs = [0.0 for _ in range(k)]
        self.timesteps = [0 for _ in range(k)]


        self.training = training
        self.testing = testing
        self.attributes = attributes
        self.weights_path = weights_path
        self.debug = debug

        if self.debug:
            print('Population:', self.population)
            print('Hyperparams:', self.hyperparams)
            print('Perfs:', self.perfs)
            print('Timesteps:', self.timesteps)


    # TODO: test
    def generate_net(self) -> tuple(ANN, dict):
        '''
        Generate a new network
        '''
        h = {
            'k_fold': random.randint(2, 10),
            'learning_rate': random.uniform(1e-4, 0.1),
            'momentum': random.uniform(0.0, 0.9),
            'decay': random.uniform(0.0, .01),
            'epochs': random.randint(1, 200),
            'hidden_units': [
                random.randint(1, 10) 
                for _ in range(random.randint(1, 6))
            ] # list of number of nodes in each layer
        }

        net = ANN(hyperparams=h, 
            training=self.training,
            testing=self.testing,
            attributes=self.attributes,
            weights_path=self.weights_path,
            debug=self.debug)

        return net, h
       

    def generate_population(self, population_size: int):
        '''
        Generate the population of neural networks
        '''
        for n in range(population_size):
            net, h = self.generate_net()
            self.population[n] = net
            self.hyperparams[n] = h



    def step(self, net: ANN, hyperparams: list) -> ANN:
        '''
        Apply one optimization step to the network,
        given the hyperparameters
        '''

        pass 

    def evaluate(self, net: ANN) -> float:
        '''
        Evaluate the performance of the network
        '''
        pass


    def exploit(self, net: ANN, hyperparams: list, perf: float, population: list) -> tuple(ANN, dict):
        '''
        Exploit the rest of the population 
        to find a better solution
        '''
        pass

    def explore(self, net: ANN, hyperparams: list, population: list) -> ANN:
        '''
        Produce new hyperparameters to explore
        '''
        pass

    def is_ready(self, perf: float, timestep: int, population: list) -> bool:
        '''
        Check if the net is ready to exploit
        '''
        pass    

    def is_diff(sel, net1, net2) -> bool:
        '''
        Check if the networks are different
        '''
        pass


    def train(self) -> None:
        '''
        Train the network population
        '''
        for i in range(self.k):
            net = self.population[i]
            hyperparams = self.hyperparams[i]
            perf = self.perfs[i]
            timestep = self.timesteps[i]

            while not net.end_training:
                net = self.step(net, hyperparams)
                perf = self.evaluate(net)

                if self.is_ready(perf, timestep, self.population):
                    new_net, new_hyperparams = self.exploit(net, hyperparams, perf, self.population)
                    # check if the new network is different
                    if self.is_diff(new_net, net):
                        net, hyperparams = self.explore(new_net, new_hyperparams, self.population)
                        perf = self.evaluate(net)

                # update the population
                self.population[i] = net
                self.hyperparams[i] = hyperparams
                self.perf[i] = perf
                self.timestep[i] = timestep + 1
            

        # return net with the best performance
        best_perf = max(self.perfs)
        best_net = self.population[self.perfs.index(best_perf)]

        return best_net
        
