

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

* `training.py`: gathers utility functions and classes used to train the network with the custom framework

