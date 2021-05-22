

# DL Project 2 - Automatic Differentiation

## About

Framework built on top of PyTorch that implements backpropagation, 
without using `torch.autograd`. 

## Project structure

The project is subdivided in the following Python modules:

* `tensor.py`: implements the `Tensor` class, being a wrapper of `torch.tensor`. 
  It overloads basic operations (`__add__`, `__mul__`, ...) and has the following attributes:
    * `data`: a `torch.tensor` storing actual data
    * `grad`: a `torch.tensor` of the same shape as `data` storing the gradient
    * `parents`: list of nodes of the computational graph that created the current tensor 
    * `backward_fun`: a function which takes no parameters and returns nothing, it accumulates the gradient in 
      parent nodes
  
  and has two key methods:
    * `zero_grad()`: clears the `parents` list and sets `grad` to zero
    * `backward()`: initializes the gradient to 1.0 and walk backward through the graph
    
* `module.py`: implements the following classes:
    * `Module`: superclass of all operations/functions/layers
    * `Layer`: superclass of all layers
    * `LinearLayer`: implement a linear layer of a neural network
  
* `function.py`: implements all functions, e.g. addition, matrix multiplication, relu, tanh

* `training.py`: gathers utility functions and classes used to train the network with the custom framework, 
  in particular,
    * `Dataset`: used to iterate over samples
    * `train_SGD`: train a model, given a dataset, a loss function, a learning rate and the number of epochs


## Test script

The file `test.py` generates a dataset of 10,000 samples uniformly distributed over [0,1]^2. 
Labels have a radial dependency. 80% of samples are used for the training set, 
the remaining 20% for the test set.

The model is a neural network with 3 hidden layers of 25 neurons each.
The model is trained with MSE loss function, using SGD.
