############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Assignment: Term Paper
#   Date: 4/10/2022
#   file: main.py
#   Description: main file to run the program
#############################################################

# imports
from apbt import APBT


def main():
    '''main of the program'''

    training_path = 'data/identity/identity-train.txt'
    testing_path = 'data/identity/identity-train.txt'
    attributes_path = 'data/identity/identity-attr.txt'
    debugging = True
    k = 40 # population size, 80 good number
    epochs = 20000 # number of epochs


    apbt = APBT(
        k,
        epochs,
        training_path,
        testing_path,
        attributes_path,
        debugging
    )

    # set log path
    apbt.log_path='logs/identity.csv'


    # run trials
    for trial in range(3):
        print('Trial: ', trial)
        print('\nRunning the population based training\n')
        best_net, most_acc = apbt.train()
        print('\nPopulation Based Training complete\n')
        # create the artificial neural network

        # printing the neural network
        print('\nPrinting learned weights of best\n')
        best_net.print_network()
        # test the artificial neural network
        print('\nTesting the NN...\n')
        accuracy = 100 * best_net.test(apbt.testing)
        n_params = best_net.num_params()
        print('\nTesting complete\n')
        print(f'\nAccuracy: {accuracy:.2f}%\n')
        print(f'Number of parameters: {n_params}\n')

        # printing the neural network
        print('\nPrinting learned weights of the most accurate\n')
        most_acc.print_network()
        # test the artificial neural network
        print('\nTesting the NN...\n')
        accuracy = 100 * most_acc.test(apbt.testing)
        n_params = most_acc.num_params()
        print('\nTesting complete\n')
        print(f'\nAccuracy: {accuracy:.2f}%\n')
        print(f'Number of parameters: {n_params}\n')


    
if __name__ == '__main__':
    main()
