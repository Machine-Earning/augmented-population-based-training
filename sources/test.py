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

    training_path = 'data/tennis/tennis-train.txt'
    testing_path = 'data/tennis/tennis-test.txt'
    attributes_path = 'data/tennis/tennis-attr.txt'
    weights_path = 'models/weights.txt'
    debugging = True
    k = 10
    epochs = 500


    apbt = APBT(
        k,
        epochs,
        training_path,
        testing_path,
        attributes_path,
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
    accuracy = 100 * apbt.evaluate(best_net)
    print('\nTesting complete\n')
    print(f'\nAccuracy: {accuracy:.2f}%\n')


    
if __name__ == '__main__':
    main()
