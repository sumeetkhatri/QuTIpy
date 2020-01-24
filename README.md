# QuTIPy
Quantum Theory of Information for Python; pronounced "cutie pie". A package for performing calculations with quantum states and channels.


## Requirements

The code requires Python 3, and apart from the standard numpy and scipy packages, it requires cvxpy if you want to run SDPs (e.g., for the diamond norm).


## Examples

We start by importing the package:

```python
>>> from qutipy import *
```

### Creating basis vectors

To create the qubit state |0>, we execute the following line.

```python
>>> ket(2,0)
```

The first argument specifies the dimension, in this case two, and the second argument is the index for the basis vector that we want. The output of the above line is the following numpy matrix object:

```python
matrix([[1.],
        [0.]])
```

Similarly,
```python
>>> ket(2,1)
```
gives the following output:

```
matrix([[0.],
        [1.]])
```

In general, ```ket(d,i)```, for ```i``` between 0 and d-1, generates a d-dimensional column vector (as a numpy matrix) in which the ith entry contains a one.



### Taking the partial trace



### Calculating von Neumann entropy



## FAQ

**1. How is this package different from qutip?**

**2. Does the package support symbolic computations?**
