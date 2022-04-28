# Augmented Population Based Training

This is an implementation of augmented population-based training. The necessary files are the following 
    main.py            (main file with options)
    ann.py             (dependency for all neural net functionalities)
    apbt.py            (dependency for all augmented population-based training functionalities)
    utils.py           (dependency for utility functions used)
    testIdentity.py    (main identity experiment file)
    testTennis.py      (main tennis experiment file)
    testIris.py        (main Iris experiment file)
    data/              (directory of all required attribute, training and testing data files)

No need to compile since the Python files are interpreted

To run the tree with options, use the following command example:

```
$ python source/main.py \
-a data/iris/iris-attr.txt \
-d data/iris/iris-train.txt \
-t data/iris/iris-test.txt \
-w models/weights.txt \
-k 80 \
-e 3000 \
--debug
```

where python3 is the python 3.X.X interpreter, 
usage: main.py [-h] -a ATTRIBUTES -d TRAINING -t TESTING [-w WEIGHTS] -k K_INDS -e EPOCHS
               [--debug]
        Population Based Training for Artificial Neural Networks

        optional arguments:
        -h, --help            show this help message and exit
        -a ATTRIBUTES, --attributes ATTRIBUTES
                                path to the attributes files (required)
        -d TRAINING, --training TRAINING
                                path to the training data files (required)
        -t TESTING, --testing TESTING
                                path to the test data files (optional)
        -w WEIGHTS, --weights WEIGHTS
                                path to save the weights (optional)
        -k K_INDS, --k-inds K_INDS
                                number of individuals in the population (default: 60)
        -e EPOCHS, --epochs EPOCHS
                                number of epochs to train
        --debug               debug mode, prints statements activated (optional)

To find out about the options, use:
```
$ python3 main.py -h 
```
To run the different experiment files (should be in same directory as data files), use the following  command:
```
$ python3 testIdentity.py
$ python3 testTennis.py 
$ python3 testIris.py
```
where python3 is the python 3.X.X interpreter, and provided the data files are present 
and in the same directory as the experiment files




