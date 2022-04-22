# Natural gradient based DSNGD in large dimension manifold 

This project implements a model for the classification problem
where a variable Y
is desired to be predicted after a variable X, by optimizing the 
log likelihhood function or the conditional Kullback-Leibler divergence. 
Implementation of optimization algorithm dsngd added as well as 
sgd and adagrad (adding more algorithms in the future).
The code found in this project is used to create the graph comparing
DSNGD to SGD and AdaGrad for my Ph.D Thesis with title: _Efficient 
and convergent natural gradient based optimization 
algorithms for machine learning_ 
### Running the default experiment
Clone the project, access to the directory in your command line and 
execute the _experiment.py_ coding file:
```bash
python3 experiment.py
```
### Set up a new experiment
For a custom experiment with different settings open the _experiemnt.py_ file
and fill the variables with the desired values. Modifiable variables are:

```python
## Manifold related variables
y_values = 10  # Classes of discrete variable Y
xd_values = [7,6,7,2,7]  # Values of discrete variables  x_i in X assuming Naive Bayes
xg_values = 0  # Amount of x_i gaussian variables in X assuming Naive Bayes

## Algorithm related variables
algs = [sgd,  adagrad, dsngd] # A list of the algorithms to test
batch = 1  # Batch of sample fed to algorithm per iteration

## Sample related variables
sample_length = 100000  # Length of the sample
epochs = 1  # Repetitions of the sample
```