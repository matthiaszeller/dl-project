# DeepLearning Project 1 - MNIST Digit Comparison


## About

Implementation of Weight Sharing and Auxiliary Loss on PyTorch for digit comparison from MNIST Dataset. 


## Running
Run `test.py` without arguments to run each model once, displaying training and test accuracy. 

**Note** : If the folder `data` of the MNIST Database is already present in the root `Proj1`, 
then the dataset won't be downloaded.


## Project structure

The project is subdivided in the following Python modules:


* `test.py`: Runs the models and print accuracies

* `models.py`: Implements models that inherit from `torch.nn.Module`

* `train.py`: Implements utility functions for training and testing

* `stats.py`: Implements tools to get statistics from running models multiple times, makes plots

* `utils.py`: The script provided by the teacher. Includes tools to download MNIST data

