This is an implementation of neural network with support for momentum, weight decay, and k-fold crossvalidation. The necessary files are the following 
    main.py            (main file with options)
    ann.py             (dependency for all neural net functionalities)
    utils.py           (dependency for utility functions used)
    testIdentity.py    (main identity experiment file)
    testTennis.py      (main tennis experiment file)
    testIris.py        (main Iris experiment file)
    testIrisNoisy.py   (main Iris with Noise experiment file)
    data/              (directory of all required attribute, training and testing data files)

No need to compile since the Python files are interpreted

To run the tree with options, use the following command example:

$ python source/main.py \
-a data/identity/identity-attr.txt \
-d data/identity/identity-train.txt \
-t data/identity/identity-train.txt \
-k 10 \
-w models/weights.txt \
-u 3 \
-e 5000 \
-l 1e-3 \
-m 0.0 \
-g 0.0 \
--debug

where python3 is the python 3.X.X interpreter, 
    optional arguments:
    -a ATTRIBUTES, --attributes ATTRIBUTES
                            path to the attributes files (required)
    -d TRAINING, --training TRAINING
                            path to the training data files (required)
    -t TESTING, --testing TESTING
                            path to the test data files (required)
    -k K_FOLD, --k-fold K_FOLD
                            number of folds for k-fold cross validation, k=0 or k=1 for no validation
    -w WEIGHTS, --weights WEIGHTS
                            path to save the weights (optional)
    -u HIDDEN_UNITS, --hidden-units HIDDEN_UNITS
                            number of hidden units (default: 3)
    -e EPOCHS, --epochs EPOCHS
                            number of epochs (default: 10)
    -l LEARNING_RATE, --learning-rate LEARNING_RATE
                            learning rate (default: 0.01)
    -m MOMENTUM, --momentum MOMENTUM
                            momentum (default: 0.9)
    -g DECAY, --decay DECAY
                            weight decay gamma (default: 0.01)
    --debug               debug mode, prints statements activated (optional)
To find out about the options, use:
$ python3 main.py -h 

To run the different experiment files, use the following  command:

$ python3 testIdentity.py
$ python3 testTennis.py 
$ python3 testIris.py
$ python3 testIrisNoisy.py

where python3 is the python 3.X.X interpreter, and provided the data files are present 
and in the same directory as the experiment files




