# QuTIPy
Quantum Theory of Information for Python; pronounced "cutie pie". A package for performing calculations with quantum states and channels. It it similar to the [QETLAB package](http://www.qetlab.com/Main_Page) for MATLAB/Octave.


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

```python
matrix([[0.],
        [1.]])
```

In general, ```ket(d,j)```, for ```j``` between ```0``` and ```d-1```, generates a d-dimensional column vector (as a numpy matrix) in which the jth entry contains a one.

We can take tensor products of d-dimensional basis vectors using ```ket()```. For example, the two-qubit state |0>|0> can be created as follows:

```python
>>> ket(2,[0,0])
```

In general, ```ket(d,[j_1,j_2,...,j_n])``` creates the n-fold tensor product |j_1>|j_2>...|j_n> of d-dimensional basis vectors.


### Taking the partial trace

Given an operator ```R_AB``` acting on a tensor product Hilbert space of the quantum systems ```A``` and ```B```, the partial trace over ```B``` can be calculated as follows:

```python
>>> TrX(R_AB,[2],[dimA,dimB])
```

Here, ```dimA``` is the dimension of system ```A``` and ```dimB``` is the dimension of system ```B```. Similarly,

```python
>>> TrX(R_AB,[1],[dimA,dimB])
```

takes the partial trace of ```R_AB``` over system ```A```. In general, ```TrX(R,sys,dim)``` traces over the systems in the list ```sys```, and ```dim``` is a list of the dimensions of all of the subsystems on which the operator ```R``` acts.



## FAQ

[comment]**1. How is this package different from qutip?**

[comment]**2. Does the package support symbolic computations?**

## Acknowledgements

Thanks to Mark Wilde for suggesting the name for the package.



