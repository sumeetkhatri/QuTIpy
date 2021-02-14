# QuTIpy
Quantum Theory of Information for Python; pronounced "cutie pie". A package for performing calculations with quantum states and channels. It is comparable to the [QETLAB package](http://www.qetlab.com/Main_Page) for MATLAB/Octave.


## Requirements

The code requires Python 3, and apart from the standard numpy and scipy packages, it requires cvxpy if you want to run SDPs (e.g., for the diamond norm). It requires sympy for symbolic computations.


## Examples

Here are some simple examples.

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
>>> partial_trace(R_AB,[2],[dimA,dimB])
```

Here, ```dimA``` is the dimension of system ```A``` and ```dimB``` is the dimension of system ```B```. Similarly,

```python
>>> partial_trace(R_AB,[1],[dimA,dimB])
```

takes the partial trace of ```R_AB``` over system ```A```. In general, ```partial_trace(R,sys,dim)``` traces over the systems in the list ```sys```, and ```dim``` is a list of the dimensions of all of the subsystems on which the operator ```R``` acts.

### Quantum states

We can generate a random quantum state (i.e., density matrix) in ```d``` dimensions as follows:

```python
>>> RandomDensityMatrix(d)
```
To generate a random pure state (i.e., state vector) in ```d``` dimensions:

```python
>>> RandomPureState(d)
```

To generate an isotropic state in ```d``` dimensions:

```python
>>> isotropic_state(p,d)
```

where ```p``` is the fidelity to the maximally entangled state.

Another special class of states is the Werner states:

```python
>>> Werner_state(p,d)
```

### Quantum channels

The package comes with functions for commonly-used channels such as the depolarizing channel and the amplitude damping channel. One can also create an arbitrary qubit Pauli channel as follows:

```python
>>> Pauli_channel(px,py,pz)
```

where ```px,py,pz``` are the probabilities of the individual pauli matrices. The output of this function contains the Kraus operators of the channel as well as an isometric extension of the channel.

In order to apply a quantum channel to a quantum state ```rho```, we can use the function ```apply_channel```. First, let us define the following amplitude damping channel:

```python
>>> K=amplitude_damping_channel(0.2)
```

The variable ```K``` contains the Kraus operators of the channel. Then,

```python
>>> rho_out=apply_channel(K,rho)
```

gives the state at the output of the channel when the input state is ```rho```.

Other functions include:

- Getting the Choi and natural representation of a channel from its Kraus representation
- Converting between the Choi, natural, and Kraus representations of a channel


### Summary of other features

The package also contains functions for:
- Trace norm
- Fidelity and entanglement fidelity
- Random unitaries
- Clifford unitaries
- Generators of the su(d) Lie algebra (for d=2, this is the set of Pauli matrices)
- Discrete Weyl operators
- von Neumann entropy and relative entropy
- Renyi entropies
- Coherent information and Holevo information for states and channels




## Acknowledgements

Thanks to Mark Wilde for suggesting the name for the package.



