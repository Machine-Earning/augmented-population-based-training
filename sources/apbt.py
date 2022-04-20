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
        debug) -> None:
        '''
        Initialize the APBT class
        '''
        self.k = k
        self.population = [None for _ in range(k)]
        self.hyperparams = [None for _ in range(k)]
        self.perfs = [0.0 for _ in range(k)]
        self.timesteps = [0 for _ in range(k)]
        self.debug = debug

        # reading attributes 
        self.attributes, self.in_attr, self.out_attr = self.read_attributes(attributes) 
        # reading input,output lenght
        self.input_units, self.output_units = self.get_input_output_len()
        # reading data
        self.training = self.read_data(training)
        self.testing = self.read_data(testing)
        self.n_examples = len(self.training)

        if self.debug:
            print('Population:', self.population)
            print('Hyperparams:', self.hyperparams)
            print('Perfs:', self.perfs)
            print('Timesteps:', self.timesteps)
            print('Training:', self.training)
            print('Testing:', self.testing)
            print('Number of examples:', self.n_examples)


    def read_attributes(self, attr_path):
        '''
        Read in the attributes
        '''
        attributes = {}
        in_attr, out_attr = [], []
        is_input = True

            # read in the attributes
        with open(attr_path, 'r') as f:
            for line in f:
                if len(line) > 1:
                    words = line.strip().split()
                    
                    # storing the attributes
                    attributes[words[0]] = words[1:]

                    # storing the input attributes
                    if is_input:
                        in_attr.append(words[0])
                    else:
                        out_attr.append(words[0])
                    # order.append(words[0])
                else:
                    is_input = False

        if self.debug:
            print('Attributes: ', attributes)
            print('Input attributes: ', in_attr)
            print('Output attributes: ', out_attr)

        if len(attributes) == 0:
            raise Exception('No attributes found')


        return attributes, in_attr, out_attr

    def to_encode(self, attr):
        '''
        Return true if the value is discrete
        to encode
        '''
        values = self.attributes[attr]

        if len(values) > 1:
            if values[0] == '0' and values[1] == '1':
                return False
            else:
                return True
        else:
            return False

    def onehot(self, attr, value):
        '''
        Preprocess to convert a data instance 
        to one-of-n/onehot encoding
        '''
        # get the index of the value
        encoded = [0.0 for _ in range(len(self.attributes[attr]))]
        encoded[self.attributes[attr].index(value)] = 1.0

        if self.debug:
            print('One-hot encoded: ', encoded)

        return encoded

    def read_data(self, data_path):
        '''
        Read in the training data and testing data
        '''
        data = []

        # read in the attributes
        with open(data_path, 'r') as f:
            for line in f:
                if len(line) > 0:
                    items = line.strip().split()
                    # get items iterator
                    items_iter = iter(items)

                    In, Out = [],[]
                    # get inputs
                    for attr in self.in_attr:
                        value = next(items_iter)
                        if self.to_encode(attr):
                            # encode discrete values
                            encoded = self.onehot(attr, value)
                            In += encoded # since encoded is a list
                        else:
                            # encode continuous values
                            In.append(float(value))

                    # get outputs
                    for attr in self.out_attr:
                        value = next(items_iter)
                        if self.to_encode(attr):
                            # encode discrete values
                            encoded = self.onehot(attr, value)
                            Out += encoded # since encoded is a list
                        else:
                            # encode continuous values
                            Out.append(float(value))

                    # check if the encoding should be applied
                    # when encoding applied, update the input or output units sizes
                    data.append([In, Out])

        if len(data) == 0:
            raise Exception('No data found')

        return data

    def decode(self, attr, encoded):
        '''
        Decode the encoded value
        '''
        # get the index of the value
        # value = self.attributes[attr][encoded.index(1.0)]
        if self.debug:
            print('Encoded: ', encoded)
            print('attr: ', attr)
            print('Attributes: ', self.attributes[attr])
        value_encoded = zip(self.attributes[attr], encoded)
        # sort the encoded value
        sorted_encoded = sorted(value_encoded, key=lambda x: x[1], reverse=True)

        # get the value
        value = sorted_encoded[0][0]

        if self.debug:
            print('Decoded: ', value)
            print('Sorted encoded: ', sorted_encoded)

        return value

    def get_input_output_len(self):
        '''
        Get the input and output units
        '''
        # getting total number of input units
        input_units = 0
        for attr in self.in_attr:
            values = self.attributes[attr]
            # check specifically for identity
            if values[0] == '0' and values[1] == '1':
                input_units += 1
            else:
                input_units += len(values)

        # getting total number of output units  
        output_units = 0
        for attr in self.out_attr:
            values = self.attributes[attr]
            # check specifically for identity
            if values[0] == '0' and values[1] == '1':
                output_units += 1
            else:
                output_units += len(values)

        return input_units, output_units

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

        net = ANN(
            hyperparams=h, 
            input_units=self.input_units,
            output_units=self.output_units,
            debug=self.debug
        )

        return net, h
       

    def generate_population(self, population_size: int):
        '''
        Generate the population of neural networks
        '''
        for n in range(population_size):
            net, h = self.generate_net()
            self.population[n] = net
            self.hyperparams[n] = h


    # TODO: implement
    def step(self, net: ANN, hyperparams: dict) -> ANN:
        '''
        Apply one optimization step to the network,
        given the hyperparameters
        '''
        net.set_hyperparameters(hyperparams)
        net.step()
        return net

    # TODO: test
    def evaluate(self, net: ANN) -> float:
        '''
        Evaluate the performance of the network
        '''
        for n in range(self.k):
            net = self.population[n]
            perf = net.test()
            self.perfs[n] = perf

    # TODO: implement 
    def exploit(self, net: ANN, hyperparams: list, perf: float, population: list) -> tuple(ANN, dict):
        '''
        Exploit the rest of the population 
        to find a better solution
        '''
        pass

    # TODO: implement
    def explore(self, net: ANN, hyperparams: list, population: list) -> ANN:
        '''
        Produce new hyperparameters to explore
        '''
        pass

    # TODO: implement
    def is_ready(self, perf: float, timestep: int, population: list) -> bool:
        '''
        Check if the net is ready to exploit
        '''
        pass    

    # TODO: test
    def is_diff(sel, net1, net2) -> bool:
        '''
        Check if the networks are different
        '''
        # check if the weights are different
        return net1.weights != net2.weights


    # TODO: test
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
        
