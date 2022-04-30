############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Assginment: Term Paper
#   Date: 4/10/2022
#   file: ann.py
#   Description: main class for artificial neural network
#   Implementation of backpropagation algorithm for a 
#   feed forward neural network
#   input parameters are:
#   - learning rate
#   - momentum
#   - weight decay
#   - topology of the network: encoded as
#   a list of integers where each integer represents the
#   number of nodes in the respective layer
#############################################################

import math
import random

class ANN:
    '''
    Feed Forward Artificial Neural Network Class
    1 Input, 1 Hidden, 1 Output Layer
    '''
    def __init__(
        self, 
        net_id,
        hyperparams,
        input_units, 
        output_units, 
        debug=True,
    ):
        
        '''
        Initialize the Artificial Neural Network
        '''
        # hyperparameters
        self.net_id = net_id
        self.hidden_units = hyperparams['hidden_units']
        self.learning_rate = hyperparams['learning_rate']
        self.momentum = hyperparams['momentum']
        self.decay = hyperparams['decay']
        self.input_units = input_units
        self.output_units = output_units
        self.debug = debug

        # initialize the weights at random based 
        # the topology of the network
        self.topology = [self.input_units] + \
                        hyperparams['hidden_units'] + \
                        [self.output_units]

        self.weights = {
            f'W{i}{i-1}': [[self.rand_init() 
                        for _ in range(self.topology[i-1] + 1)]
                    for _ in range(self.topology[i])] 
                for i in range(1, len(self.topology))
        }

        self.res = None
        # self.num_params = self.num_params()

        # print the everything
        # if self.debug:
        #     print('learning rate: ', self.learning_rate)
        #     print('momentum: ', self.momentum)
        #     print('Weights: ', self.weights)
        #     print('Topology: ', self.topology)

    def rand_init(self):
        '''
        Initialize the weights at random
        https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
        '''
        # number of nodes in the previous layer
        n = self.input_units
        # calculate the range for the weights
        lower, upper = -(1.0 / math.sqrt(n)), (1.0 / math.sqrt(n))
        # generate random numbers
        number = random.random()
        # scale to the desired range
        scaled = lower + number * (upper - lower)

        return scaled
             
    def print_weights(self):
        '''
        Print the weights of the Artificial Neural Network
        with 2 decimal places
        '''
        # print('Weights: ', self.weights)
        for key, value in self.weights.items():
            print(f'{key}: ')
            for i in range(len(value)):
                print(f'{i}: ', end='')
                for j in range(len(value[i])):
                    print(f'{value[i][j]:.2f} ', end='')
                print()

    def print_network(self):
        '''
        Print the network
        '''
        print('Network: ', self.topology)
        self.print_weights()

    def set_hyperparameters(self, hyperparams):
        '''
        Set the hyperparameters of the Artificial Neural Network
        '''
        self.learning_rate = hyperparams['learning_rate']
        self.momentum = hyperparams['momentum']
        self.decay = hyperparams['decay']
        self.hidden_units = hyperparams['hidden_units']
        self.topology = [self.input_units] + \
            hyperparams['hidden_units'] + \
            [self.output_units]
    
    def num_params(self):
        '''
        Get the number of parameters
        '''
        num_parameters = 0
        for t in range(1, len(self.topology)):
            num_parameters += self.topology[t] * self.topology[t-1]
        return num_parameters

    def save(self, filename=None):
        '''
        Save the Artificial Neural Network
        '''
        # if no filename is provided, then use the default
        filename = filename or 'weights.txt'
        # save the weights onto a file
        with open(filename, 'w') as f:
            f.write(str(self.weights))

    def load(self, filename):
        '''
        Load the Artificial Neural Network
        '''
        # load the weights from a file
        with open(filename, 'r') as f:
            self.weights = eval(f.read())

        # print('one weight: ', self.weights['hidden'][0][0])
        self.print_weights()

    def get_classes(self):
        '''
        Get the output classes
        '''
        
        num_out = self.output_units
        classes = [
            [0.0 for _ in range(num_out)] 
            for _ in range(num_out)
        ]

        for i in range(num_out):
            classes[i][i] = 1.0
        
        return classes

    def sigmoid(self, x):
        '''
        Sigmoid activation function
        '''
        # print('bad x:', x)
        try : return 1 / (1 + math.exp(-x))
        except OverflowError: return 0.0

    def d_sigmoid(self, x):
        '''
        Derivative of the sigmoid function
        '''
        y = self.sigmoid(x)
        return y * (1 - y)

    def forward(self, instance):
        '''
        Feed forward the Artificial Neural Network
        '''

        res = {
            f'layer{i}': [0.0 for _ in range(self.topology[i])]
                for i in range(1, len(self.topology))
        }

        # set the input layer to the instance
        res['layer0'] = instance

        # feed forward the hidden layer
        for t in range(1, len(self.topology)):
            for i in range(self.topology[t]):
                for j in range(self.topology[t-1]):
                    # calculating ther linear combination
                    res[f'layer{t}'][i] += \
                        self.weights[f'W{t}{t-1}'][i][j] * res[f'layer{t-1}'][j]
                # adding the bias
                res[f'layer{t}'][i] += \
                    self.weights[f'W{t}{t-1}'][i][self.topology[t-1]]

                # applying the activation function
                res[f'layer{t}'][i] = self.sigmoid(res[f'layer{t}'][i])

        # getting the output
        output = res[f'layer{len(self.topology)-1}']
        self.res = res

        return output

    def predict(self, instance):
        '''
        Predict the output of the instance
        later update to include processing of discrete input 
        and output
        '''
        return self.forward(instance)

    def loss(self, target, output, no_decay=False):
        '''
        Compute the loss for SGD
        '''
        loss = 0.0
        # getting all the loss
        for i in range(self.output_units):
            loss += (target[i] - output[i]) ** 2
        loss /= 2.0

        if no_decay: return loss 

        w_term = 0.0
        # adding all the weights
        for l in range(1, len(self.topology)):
            for i in range(len(self.weights[f'W{l}{l-1}'])):
                for j in range(len(self.weights[f'W{l}{l-1}'][i])):
                    w_term += self.weights[f'W{l}{l-1}'][i][j] ** 2

        n = self.num_params()
        loss += self.decay * (w_term / (n * 2))

        return loss

    def backward(self, target, output):
        '''
        Back propagate the error with momentum, weight 
        decay and learning rate
        SGD
        '''
        # prior delta update
        errors = {
            f'layer{l}': [0.0 for _ in range(self.topology[l])]
                for l in range(1, len(self.topology))
        }

        # calculate the errors
        for l in range(len(self.topology)-1, 0, -1):
            for i in range(self.topology[l]):
                if l == len(self.topology)-1:
                    errors[f'layer{l}'][i] = target[i] - output[i]
                else:
                    for j in range(self.topology[l+1]):
                        errors[f'layer{l}'][i] += errors[f'layer{l+1}'][j] * \
                            self.weights[f'W{l+1}{l}'][j][i]
                # applying the activation function derivative
                errors[f'layer{l}'][i] *= self.d_sigmoid(self.res[f'layer{l}'][i])

        return errors

    def step(self, errors):
        '''
        Update the weights with the errors
        '''
        deltas = {
            f'W{i}{i-1}': [[ 0.0 for _ in range(self.topology[i-1] + 1)]
                    for _ in range(self.topology[i])] 
                for i in range(1, len(self.topology))
        }
        # update the weights
        for t in range(1, len(self.topology)):
            for i in range(self.topology[t]):
                for j in range(self.topology[t-1]):
                    # update the weights
                    deltas[f'W{t}{t-1}'][i][j] = self.learning_rate * errors[f'layer{t}'][i] * \
                        self.res[f'layer{t-1}'][j] + self.momentum * deltas[f'W{t}{t-1}'][i][j]
                    self.weights[f'W{t}{t-1}'][i][j] = (1 - self.learning_rate * self.decay) * \
                        self.weights[f'W{t}{t-1}'][i][j] + deltas[f'W{t}{t-1}'][i][j]
                
                # update the bias
                deltas[f'W{t}{t-1}'][i][self.topology[t-1]] = self.learning_rate * errors[f'layer{t}'][i] \
                                    + self.momentum * deltas[f'W{t}{t-1}'][i][self.topology[t-1]]
                self.weights[f'W{t}{t-1}'][i][self.topology[t-1]] = (1 - self.learning_rate * self.decay) * \
                    self.weights[f'W{t}{t-1}'][i][self.topology[t-1]] + deltas[f'W{t}{t-1}'][i][self.topology[t-1]]

    def training_step(self, train_data):
        '''
        Train the Artificial Neural Network
        k is the number of folds
        '''
        if not train_data:
            raise ValueError('No training data provided')
        # get the data
        data = train_data
        # shuffle the data
        random.shuffle(data)
        # train the network
        loss = 0.0
        # if self.debug:
        #     print('Epoch: ', i, end='')
        for example in data:
            # loss += self.step(instance)
            inputt, target = example[0], example[1]
            # get the output
            output = self.forward(inputt)
            # compute the loss
            loss += self.loss(target, output)
            # backpropagate the errors
            errors = self.backward(target, output)
            # update the weights
            self.step(errors)
            
        # if self.debug:
        #     # print('Weights: ', self.weights)
        #     print(f'Net #{self.net_id}\'s loss: {loss/len(data): .3f}', end='')

        return loss/len(data)

    def train(self, train_data, epochs=10):
        '''
        Train the Artificial Neural Network
        k is the number of folds
        '''
        if not train_data:
            raise ValueError('No training data provided')
        # get the data
        data = train_data
        # shuffle the data
        random.shuffle(data)
        # train the network
        for _ in range(epochs):
            # if self.debug:
            #     print('Epoch: ', i, end='')
            loss = 0.0
            for example in data:
                # loss += self.step(instance)
                inputt, target = example[0], example[1]
                # get the output
                output = self.forward(inputt)
                # compute the loss
                loss += self.loss(target, output)
                # backpropagate the errors
                errors = self.backward(target, output)
                # update the weights
                self.step(errors)
            
            if self.debug:
                # print('Weights: ', self.weights)
                print(f'Net #{self.net_id}\'s loss: {loss/len(data)}', end='\n')

        return loss/len(data)

    def train_with_folds(self, train_data, epochs=1000, k=5):
        '''
        Train the Artificial Neural Network
        k is the number of folds
        '''
        if not train_data:
            raise ValueError('No training data provided')
        # get the data
        data = train_data
        # shuffle the data
        if k < 2:
            raise ValueError('k must be greater than 1')

        # randomly shuffle the data into folds 
        random.shuffle(data)
        
        # get the number of instances
        num_instances = len(data)

        while num_instances < k:
            data = data * k
            # get the number of instances
            num_instances = len(data)

        # get number of data per fold
        fold_size = num_instances // k

        # get the folds
        folds = [
            data[i:i+fold_size] for i in range(0, num_instances, fold_size)
        ]

        # scores for each fold
        scores = []
        iterations = []

        for i in range(k):

            if self.debug:
                print('Fold: ', i)

            # get the test fold
            test_fold = folds[i]
            # get the train folds
            train_folds = folds[:i] + folds[i+1:]

            # merge train folds
            train_fold = []
            for fold in train_folds:
                train_fold += fold

            # get the train data
            train_data = train_fold
            # get the test data
            validation_data = test_fold

            best_validation_loss, e = float('inf'), 0

            for i in range(epochs):
                # get the loss for training data
                # train the network
                train_loss = 0.0
                validation_loss = 0.0

                if self.debug:
                    print('Epoch: ', i, end='')

                # shuffle the data
                random.shuffle(train_data)
                
                for instance in train_data:
                    # loss += self.step(instance)
                    inputt, target = instance[0], instance[1]
                    # get the output
                    output = self.forward(inputt)
                    # compute the loss
                    train_loss += self.loss(target, output)
                    # backpropagate the errors
                    errors = self.backward(target, output)
                    # update the weights
                    self.step(errors)

                for instance in validation_data:
                    # get the output
                    output = self.feed_forward(instance[0])
                    # compute the loss
                    validation_loss += self.loss(instance[1], output)

                v_loss = validation_loss/len(validation_data)
                t_loss = train_loss/len(train_data)

                if self.debug:
                    print('\tTrain Loss: ', t_loss, '\tValidation Loss: ', v_loss)

                if i == 0 or v_loss < best_validation_loss:
                    best_validation_loss, e = v_loss, i
                elif v_loss > self.OFFSET + best_validation_loss:
                    scores.append(best_validation_loss)
                    iterations.append(e)
                    break
                elif i == self.epochs - 1:
                    scores.append(best_validation_loss)
                    iterations.append(e)

        # get the average score
        avg_score = sum(scores) / len(scores)
        avg_iter = sum(iterations) / len(iterations)

        if self.debug:
            print('Average Score: ', avg_score, '\tAverage Iterations: ', avg_iter)
        
        # train net with average iterations
        for i in range(int(avg_iter)):
            loss = 0.0

            for instance in data:
                # get the output
                output = self.forward(instance[0])
                # compute the loss
                loss += self.loss(instance[1], output)
                # backpropagate the errors
                errors = self.backward(instance[1], output)
                # update the weights
                self.step(errors)

            if self.debug:
                # print('Weights: ', self.weights)
                print('Loss: ', loss/len(data), end='\n')


    def test(self, test_data=None):
        '''
        Test the Artificial Neural Network
        ''' 
        if not test_data:
            raise Exception('No test data provided')
        # if self.debug:
        #     print('Testing data: ', test_data)
        accuracy = 0.0
        # test the network
        for instance in test_data:
            # get the output
            output = self.forward(instance[0])
            # check if the output is correct
            accuracy += (1.0 - self.loss(instance[1], output, no_decay=True))
        # get the average accuracy
        accuracy /= len(test_data)

        return accuracy
 
    

        

