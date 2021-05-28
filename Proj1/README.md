# DeepLearning Project 1 - MNIST Digit Comparison


## About

Comparison of Weight Sharing and Auxiliary Loss on PyTorch for digit comparison from MNIST Dataset. 


## Running

**Note** : If the folder `data` of the MNIST Database is already present in the root `Proj1`, 
then the dataset won't be downloaded.

### Main script
Run `test.py` without arguments to run each model once, displaying training and test accuracy. 


### Cross validation, performance evaluation, plots

Run the script `generate_plots.py` in order to:

* Tune the learning rate of each model
* Evaluate the performance of each model

in both cases, each model is run 5 times.

## Project structure

The project is subdivided in the following Python modules:


* `test.py`: Runs the models and print accuracies

* `models.py`: Implements models that inherit from `torch.nn.Module`

* `train.py`: Implements utility functions for training and testing

* `stats.py`: Implements tools to get statistics from running models multiple times, makes plots

* `utils.py`: The script provided by the teacher. Includes tools to download MNIST data

