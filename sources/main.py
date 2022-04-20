############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Assignment: Term Paper
#   Date: 4/10/2022
#   file: main.py
#   Description: main file to run the program
#############################################################

# imports
import argparse
from apbt import APBT

def parse_args():
    '''parse the arguments for artificial neural network'''

    parser = argparse.ArgumentParser(
        description='Population Based Training for Artificial Neural Networks'
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
        '-w', '--weights',
        type=str , 
        required=False,
        help='path to save the weights (optional)'
    )

    parser.add_argument(
        '-k', '--k-inds',
        type=int,
        required=True,
        help='number of individuals in the population (default: 10)'
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
    k = args.k_inds

    apbt = APBT(
        k,
        training_path,
        testing_path,
        attributes_path,
        weights_path,
        debugging
    )

    print('\nRunning the population based training\n')
    best_net = apbt.train()
    print('\nPopulation Based Training complete\n')
    # create the artificial neural network
    # printing the neural network
    print('\nPrinting learned weights\n')
    best_net.print_network()
    # save the weights
    if weights_path:
        best_net.save(weights_path)
        print('weights saved to', weights_path)

    # test the artificial neural network
    print('\nTesting the NN...\n')
    best_net.test()
    print('\nTesting complete\n')


    
if __name__ == '__main__':
    main()
