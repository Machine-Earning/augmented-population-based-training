############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/23/2022
#   file: main.py
#   Description: main file to run the program
#############################################################

# imports
import argparse
from ann import ANN

def parse_args():
    '''parse the arguments for artificial neural network'''

    parser = argparse.ArgumentParser(
        description='Artificial Neural Network for classification'
    )

    parser.add_argument(
        '-a', '--attributes',
        type=str,
        required=True,
        help='path to the attributes files (required)'
    )

    parser.add_argument(
        '-d', '--training',
        type=str, 
        required=True,
        help='path to the training data files (required)'
    )
    
    parser.add_argument(
        '-t', '--testing',
        type=str , 
        required=True,
        help='path to the test data files (required)'
    )

    parser.add_argument(
        '-k', '--k-fold',
        type=int,
        required=False,
        help='number of folds for k-fold cross validation, k=0 or k=1 for no validation'
    )

    parser.add_argument(
        '-w', '--weights',
        type=str , 
        required=False,
        help='path to save the weights (optional)'
    )

    parser.add_argument(
        '-u', '--hidden-units',
        type=int, 
        required=False,
        help='number of hidden units (default: 3)'
    )

    parser.add_argument(
        '-e', '--epochs',
        type=int, 
        required=False,
        default=10,
        help='number of epochs (default: 10)'
    )

    parser.add_argument(
        '-l', '--learning-rate',
        type=float, 
        required=False,
        default=0.1,
        help='learning rate (default: 0.01)',
    )

    parser.add_argument(
        '-m', '--momentum',
        type=float, 
        required=False,
        default=0.0,
        help='momentum (default: 0.9)',
    )

    parser.add_argument(
        '-g','--decay',
        type=float, 
        required=False,
        default=0.0,
        help='weight decay gamma (default: 0.01)',
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='debug mode, prints statements activated (optional)'
    )

    # parse arguments
    args = parser.parse_args()
    return args


def main():
    '''main of the program'''
    args = parse_args() # parse arguments
    print(' args entered',args)

    training_path = args.training
    testing_path = args.testing
    attributes_path = args.attributes
    weights_path = args.weights
    debugging = args.debug
    
    # hyperparameters
    hidden_units = args.hidden_units
    epochs = args.epochs
    learning_rate = args.learning_rate
    decay = args.decay
    momentum = args.momentum
    k_folds = args.k_fold


    print('\nCreating NN with the parameters provided\n')
    # create the artificial neural network
    ann = ANN(
        training_path, # path to training data
        testing_path, # path to testing data
        attributes_path, # path to attributes
        k_folds, # whether to use validation data
        weights_path, # path to save weights
        hidden_units, # number of hidden units
        learning_rate, # learning rate
        epochs, # number of epochs, -1 for stopping based on validation
        momentum, # momentum
        decay, # weight decay gamma
        debugging # whether to print debugging statements
    )
    # printing the neural network
    ann.print_network()


    print('\nLearning the NN...\n')
    # train the artificial neural network
    ann.train()
    print('\nTraining complete\n')

    #print weights
    print('\nPrinting learned weights\n')
    ann.print_weights()

    # save the weights
    if weights_path:
        ann.save(weights_path)
        print('weights saved to', weights_path)
        # load the weights
        # ann.load(weights_path)
        # print('weights loaded from', weights_path)

    # test the artificial neural network
    print('\nTesting the NN...\n')
    ann.test()
    print('\nTesting complete\n')


    
if __name__ == '__main__':
    main()
