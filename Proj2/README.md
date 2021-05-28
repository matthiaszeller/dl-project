

# DL Project 2 - Automatic Differentiation

## About

Framework built on top of PyTorch that implements backpropagation, 
without using `torch.autograd`. Fully implemented in Python.

## Test script

### Data
The file `test.py` generates a dataset of 10,000 samples uniformly distributed over [0,1]^2. 
Labels have a radial dependency. 

The model is a neural network with 3 hidden layers of 25 neurons each.
The model is trained with MSE loss function, using SGD. There are *six models* in total: 

* Three different activation functions between hideen layers
* Either linear output or sigmoid output

### Script 
One can run three different modes of `test.py`:

* `test.py`: **standard mode** without arguments, run each model once, display training and test accuracy.
  (Also plot dataset and learning curves, but this part is commented for the final project delivery).

  
* `test.py --cv`: **hyperparameter tuning** on the learning rate. Test a grid of learning rates for each model, using 
5-fold CV. Outputs results in `cv_results.json`.
  

* `test.py --stats`: **performance evaluation**. Run each model with 5-fold CV, twice. 
  Outputs results in `test_results.json`.



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
    * `Module`: superclass of all operations/functions/layers. Key methods: `forward`, `backward`, 
      `_forward`, `_backward`, `params`
    * `Layer`: superclass of all layers
    * `LinearLayer`: implement a linear layer of a neural network
  

* `function.py`: implements all elementary operations, e.g. addition, matrix multiplication, relu, tanh, MSE loss


* `training.py`: gathers utility functions and classes used to train the network with the custom framework, 
  in particular,
    * `Dataset`: used to iterate over samples
    * `train_SGD`: train a model, given a dataset, a loss function, a learning rate and the number of epochs. 
      Log the training loss per epoch, the training accuracy per epoch and computation time per epoch.
    * `kfold_cv`: perform k-fold cross validation on a model  
