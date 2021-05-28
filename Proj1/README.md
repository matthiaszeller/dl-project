# DeepLearning Project 1 - MNIST Digit Comparison


## About

Implementation of Weight Sharing and Auxiliary Loss on PyTorch for digit comparison from MNIST Dataset. 


## Running
Run `test.py` without arguments to run each model once, displaying training and test accuracy. 

**Note** : If the folder `data` of the MNIST Database is already present in the same folder as the scripts, then they won't download it. 
## Project structure

The project is subdivided in the following Python modules:


* `test.py`: Main class, runs the models and gives feedback about them

* `models.py`: Implements models inheriting from torch.nn.Module

* `train.py`: Implements utils for training and testing

* `stats.py`: Implements tools to get stats from running models multiple times and for making plots

* `utils.py`: Includes tools to generate data from MNIST database
