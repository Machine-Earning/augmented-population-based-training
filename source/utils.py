############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Assginment: Term Paper
#   Date: 4/10/2022
#   file: utils.py
#   Description: 
#############################################################

from email import header
import random


def corrupt_data(data, classes, percent):
    '''corrupt the class labels of training examples from 0% to 20% (2% in-
    crement) by changing from the correct class to another class; output the
    accuracy on the uncorrupted test set with and without rule post-pruning.'''

    # get the number of training examples
    num_examples = len(data)
    # get the number of classes to corrupt
    num_examples_to_corrupt = int(percent * num_examples)
    # get the elements to corrupt
    corrupt_elements = random.sample(range(num_examples), num_examples_to_corrupt)

    # corrupt the data
    for e in corrupt_elements:
        # get the class label
        correct_label = data[e][-1]
        
        random_class = random.choice(classes)

        # while the random class is the same as the correct class
        while random_class == correct_label:
            random_class = random.choice(classes)
    
        # change the class label
        data[e][-1] = random_class
        
    return data

def log_csv(path, histories, headers):
    '''log the data to the csv file'''
    headers = ['e'] + headers
    # open the file
    with open(path, 'w') as f:
        # write the headers
        f.write(','.join(headers) + '\n')
        # write the data
        for h in range(len(histories[0])):
            line = f'{h},'
            for hh in range(len(histories)):
                line += str(histories[hh][h]) + ','
            f.write(line[:-1] + '\n')
    
    # epoch is number of every line
    


