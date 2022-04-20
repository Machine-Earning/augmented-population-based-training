############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Assginment: Term Paper
#   Date: 4/10/2022
#   file: ann.py
#   Description: main class for artificial neural network
#   Implementation of backpropagation algorithm for a 
#   feed forward neural network with one hidden layer
#   input parameters are:
#   - learning rate
#   - number of epochs
#   - momentum
#   - weight decay
#   - number of folds for k-fold cross validation
#   - Architecture or topology of the network: encoded as
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
        hyperparams,
        input_units, 
        output_units, 
        debug=True,
    ) -> None:
        
        '''
        Initialize the Artificial Neural Network
        '''
        
        # self.end_training = False
        # hyperparameters
        self.k_fold = hyperparams['k_fold']
        self.hidden_units = hyperparams['hidden_units']
        self.learning_rate = hyperparams['learning_rate']
        self.momentum = hyperparams['momentum']
        self.decay = hyperparams['decay']
        self.epochs = hyperparams['epochs']
        self.input_units = input_units
        self.output_units = output_units
        self.debug = debug
        # self.INIT_VAL = 0.01 # initial value for weights and biases
        self.OFFSET = .05 # offset for early stopping

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

        # print the everything
        if self.debug:
            print('learning rate: ', self.learning_rate)
            print('momentum: ', self.momentum)
            print('epochs: ', self.epochs)
            print('Weights: ', self.weights)
            print('Topology: ', self.topology)

    def rand_init(self) -> float:
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

    def set_hyperparams(self, hyperparams):
        '''
        Set the hyperparameters of the Artificial Neural Network
        '''
        self.k_fold = hyperparams['k_fold']
        self.learning_rate = hyperparams['learning_rate']
        self.momentum = hyperparams['momentum']
        self.decay = hyperparams['decay']
        self.epochs = hyperparams['epochs']
        self.hidden_units = hyperparams['hidden_units']
        self.topology = [self.input_units] + \
            hyperparams['hidden_units'] + \
            [self.output_units]
        

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
        try :
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.0
        # return 1 / (1 + math.exp(-x))

    def d_sigmoid(self, x):
        '''
        Derivative of the sigmoid function
        '''
        y = self.sigmoid(x)
        return y * (1 - y)


    # TODO: test
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

    # TODO: test
    def loss(self, target, output):
        '''
        Compute the loss for SGD
        '''
        loss = 0.0
        # getting all the loss
        for i in range(self.output_units):
            loss += (target[i] - output[i]) ** 2
        loss /= 2.0

        w_term = 0.0
        # adding all the weights
        for l in range(1, len(self.topology)):
            for i in range(len(self.weights[f'W{l}{l-1}'])):
                for j in range(len(self.weights[f'W{l}{l-1}'][i])):
                    w_term += self.weights[f'W{l}{l-1}'][i][j] ** 2

        loss += self.decay * w_term

        return loss

    # TODO: test
    def backward(self, target, output):
        '''
        Back propagate the error with momentum, weight 
        decay and learning rate
        SGD
        '''
        # if self.debug:
        #     print('inputs: ', inputs)
        #     print('Target: ', target)
        #     print('Output: ', output)

        # prior delta update
        deltas = {
            f'W{i}{i-1}': [[ 0.0 for _ in range(self.topology[i-1] + 1)]
                    for _ in range(self.topology[i])] 
                for i in range(1, len(self.topology))
        }

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

                errors[f'layer{l}'][i] *= self.d_sigmoid(self.res[f'layer{l}'][i])

        # weight update factor based on decay
        factor = 1 - 2 * self.learning_rate * self.decay

        # update the weights
        for t in range(1, len(self.topology)):
            for i in range(self.topology[t]):
                for j in range(self.topology[t-1]):
                    # update the weights
                    deltas[f'W{t}{t-1}'][i][j] = factor * errors[f'layer{t}'][i] * self.res[f'layer{t-1}'][j] \
                                    + self.momentum * deltas[f'W{t}{t-1}'][i][j]
                    self.weights[f'W{t}{t-1}'][i][j] += deltas[f'W{t}{t-1}'][i][j]
                
                # update the bias
                deltas[f'W{t}{t-1}'][i][self.topology[t-1]] = factor * errors[f'layer{l}'][i] \
                                    + self.momentum * deltas[f'W{t}{t-1}'][i][self.topology[t-1]]
                self.weights[f'W{t}{t-1}'][i][self.topology[t-1]] += deltas[f'W{t}{t-1}'][i][self.topology[t-1]]


    def step(self, example):
        '''
        Perform a single step of SGD
        '''
        inputt, target = example[0], example[1]
        output = self.forward(inputt)
        loss = self.loss(target, output)
        self.backward(target, output)

        return loss



    # TODO: added train for step
    def train(self, train_data):
        '''
        Train the Artificial Neural Network
        k is the number of folds
        '''
        if not train_data:
            raise ValueError('No training data provided')

        # get the data
        data = train_data
        # get number of folds
        k = self.k_fold

        if k > 1:
            # randomly shuffle the data into folds 
            random.shuffle(data)
            
            # get the number of instances
            num_instances = len(data)

            while num_instances < k:
                data = data * k
                # get the number of instances
                num_instances = len(data)

            # if self.debug:
            #     print('data: ', data)

            # get number of data per fold
            fold_size = num_instances // k

            # get the folds
            folds = [
                data[i:i+fold_size] 
                for i in range(0, num_instances, fold_size)
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

                # if self.debug:
                #     print('Test Fold: ', test_fold)
                #     print('Train Folds: ', train_fold)

                # get the train data
                train_data = train_fold
                # get the test data
                vali_data = test_fold

                best_vali_loss, e = float('inf'), 0

                for i in range(self.epochs):
                    
                    # get the loss for training data
                    # train the network
                    train_loss = 0.0
                    vali_loss = 0.0

                    if self.debug:
                        print('Epoch: ', i, end='')

                    # shuffle the data
                    random.shuffle(train_data)
                    
                    for instance in train_data:
                        # get the output
                        train_loss += self.step(instance)

                    # if self.debug:
                    #     # print validation data
                    #     print('validation: ', vali_data)

                    for instance in vali_data:
                        # compute the loss
                        vali_loss += self.loss(instance[1], self.forward(instance[0]))


                    v_loss = vali_loss/len(vali_data)
                    t_loss = train_loss/len(train_data)

                    if self.debug:
                        print('\tTrain Loss: ', t_loss, '\tValidation Loss: ', v_loss)

                    if i == 0 or v_loss < best_vali_loss:
                        best_vali_loss, e = v_loss, i
                    elif v_loss > self.OFFSET + best_vali_loss:
                        scores.append(best_vali_loss)
                        iterations.append(e)
                        break
                    elif i == self.epochs - 1:
                        scores.append(best_vali_loss)
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
                    # output = self.forward(instance[0])
                    # # compute the loss
                    # loss += self.loss(instance[1], output)
                    # # back propagate the error
                    # self.backward(instance, output)
                    loss += self.step(instance)

                if self.debug:
                    # print('Weights: ', self.weights)
                    print('Loss: ', loss/len(data), end='\n')

        else:
            # shuffle the data
            random.shuffle(data)
            # train the network
            for i in range(self.epochs):
                loss = 0.0

                if self.debug:
                    print('Epoch: ', i, end='')

                for instance in data:

                    # get the output
                    # output = self.forward(instance[0])
                    # # get the loss per instance
                    # loss += self.loss(instance[1], output)
                    # # update the weights
                    # self.backward(instance, output)
                    loss += self.step(instance)
                    
                if self.debug:
                    # print('Weights: ', self.weights)
                    print('Loss: ', loss/len(data), end='\n')

            # save the weights
            # if self.weights_path:
            #     self.save()


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
            # if self.debug:
            #     print('hidden results: ', self.hidden_res, end='\n')

            # if self.debug:
            #     print('Output: ', output)
            #     print('Target: ', instance[1])
            #     print('Loss: ', self.loss(instance[1], output), end='\n')
            
            # check if the output is correct
            accuracy += 1.0 - self.loss(instance[1], output)

        accuracy /= len(test_data)

        # if self.debug:
        #     print('Accuracy: ', accuracy * 100, '%')
        
        return accuracy
 
    

        

