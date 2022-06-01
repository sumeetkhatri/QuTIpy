# QuTIpy
[![CI](https://github.com/arnavdas88/QuTIpy/actions/workflows/ci.yml/badge.svg)](https://github.com/arnavdas88/QuTIpy/actions/workflows/ci.yml)

Quantum Theory of Information for Python
pronounced "cutie pie". A package for performing calculations with quantum states and channels. It is comparable to the [QETLAB package](http://www.qetlab.com/Main_Page) for MATLAB / Octave.

Read more about the QuTIpy package at our [GitBook](https://arnav-das.gitbook.io/qutipy-quantum-theory-of-information-for-python/).


## Requirements
The code requires Python 3, and apart from the standard `numpy` and `scipy` packages, it requires `cvxpy` if you want to run [SDPs](https://en.wikipedia.org/wiki/Semidefinite\_programming) (e.g., for the [diamond norm](https://en.wikipedia.org/wiki/Diamond\_norm)). It requires `sympy` for symbolic computations.

## Installation

A simple _pip install_ from the github repository will install the package in your system.

```bash
$ pip install git+https://github.com/sumeetkhatri/QuTIpy
```

## Examples

Here are some simple examples.

We start by importing the package:

```python
>>> from qutipy import *
>>> from qutipy.general_functions import *
```


### Creating basis vectors

To create the qubit [state](https://en.wikipedia.org/wiki/Quantum\_state) ${\displaystyle |0\rangle }$, we execute the following line.

```python
>>> ket(2,0)
```

The first argument specifies the dimension, in this case two, and the second argument is the index for the [basis vector](https://en.wikipedia.org/wiki/Basis\_\(linear\_algebra\)) that we want. The output of the above line is the following numpy matrix object:

```python
ndarray([[1.],
        [0.]])
```

Similarly,

```python
>>> ket(2,1)
```

gives the following output:

```python
ndarray([[0.],
        [1.]])
```

In general, `ket(d,j)`, for `j` between `0` and `d-1`, generates a d-dimensional column vector (as a numpy matrix) in which the jth entry contains a one.

We can take tensor products of d-dimensional basis vectors using `ket()`. For example, the two-qubit state ${\displaystyle |0\rangle|0\rangle }$ can be created as follows:

```python
>>> ket( 2, [0, 0] )
```

In general, `ket(d, [j1, j2, ... , jn])` creates the n-fold tensor product ${\displaystyle |j_1\rangle|j_2\rangle...|j_n\rangle }$  of d-dimensional basis vectors.


### Taking the partial trace

Given an operator $R_{AB}$ acting on a tensor product [Hilbert space](https://en.wikipedia.org/wiki/Hilbert\_space) of the quantum systems `A` and `B`, the [partial trace](https://en.wikipedia.org/wiki/Partial\_trace) over `B` can be calculated as follows:

```python
>>> partial_trace(R_AB, [2], [dimA, dimB])
```

Here, `dimA` is the dimension of system `A` and `dimB` is the dimension of system `B`. Similarly,

```python
>>> partial_trace(R_AB, [1], [dimA, dimB])
```

takes the partial trace of `R_AB` over system `A`. In general, `partial_trace(R,sys,dim)` traces over the systems in the list `sys`, and `dim` is a list of the dimensions of all of the subsystems on which the operator `R` acts.


### Quantum states

We can generate a random [quantum state](https://en.wikipedia.org/wiki/Quantum\_state#Mixed\_states) (i.e., [density matrix](https://en.wikipedia.org/wiki/Density\_matrix)) in `d` dimensions as follows:

```python
>>> RandomDensityMatrix(d)
```

To generate a random [pure state](https://en.wikipedia.org/wiki/Quantum\_state#Pure\_states) (i.e., state vector) in `d` dimensions:

```python
>>> RandomPureState(d)
```

To generate an isotropic state in `d` dimensions:

```python
>>> isotropic_state(p,d)
```

where `p` is the fidelity to the maximally entangled state.

Another special class of states is the Werner states:

```python
>>> Werner_state(p,d)
```

> The **Isotropic State** can be viewed as a probabilistic mixture of the [Qudit](https://en.wikipedia.org/wiki/Qubit#Qudits\_and\_qutrits) Bell states, such that the state ${\displaystyle |\phi\rangle\langle\phi| }$ is prepared with probability $p$, and the states ${\displaystyle |\phi_{z,x}\rangle\langle\phi_{z,x}| }$, with $(z, x) \neq (0, 0)$, are prepared with probability $\frac{1−p} {d^2−1}$. This implies that every isotropic state is a Bell-diagonal state, that it has full rank, and that its eigenvalues are $p$ and $\frac{1−p} {d^2−1}$ (the latter with multiplicity $d^2 − 1$).


> The **Werner state**  ${\displaystyle W_{AB}^{(p,d)}}$, for 2 quantum systems $A$ and $B$, with $d_A = d_B = d ≥ 2$, is a mixture of [projectors](https://en.wikipedia.org/wiki/Projection\_\(linear\_algebra\)) onto the symmetric and antisymmetric subspaces, with the relative weight ${\displaystyle p\in [0,1]}$ being the main parameter that defines the state,&#x20; for ${\displaystyle \rho_{AB} = {\rho_{AB}}^{W;p}  }$ ,&#x20;such that ${\rho_{AB}}^{W;p} := p\zeta_{AB} + (1 − p)\zeta^\bot_{AB}$

where  $\zeta_{AB}$ and $\zeta^\bot_{AB}$ are quantum states and are proportional to the projections onto the anti-symmetric and symmetric subspaces respectively.



### Quantum channels

The package comes with functions for commonly-used channels such as the depolarizing channel and the amplitude damping channel. One can also create an arbitrary [Qubit](https://en.wikipedia.org/wiki/Qubit) Pauli channel as follows:

```python
>>> Pauli_channel(px, py, pz)
```

where `px, py, pz` are the probabilities of the individual [Pauli Matrices](https://en.wikipedia.org/wiki/Pauli\_matrices). The output of this function contains the [Kraus operators](https://en.wikipedia.org/wiki/Quantum\_operation#Kraus\_operators) of the [channel](https://en.wikipedia.org/wiki/Quantum\_channel) as well as an isometric extension of the channel.

In order to apply a [quantum channel](https://en.wikipedia.org/wiki/Quantum\_channel) to a [quantum state](https://en.wikipedia.org/wiki/Quantum\_state) `rho`, we can use the function `apply_channel`. First, let us define the following [amplitude damping channel](https://en.wikipedia.org/wiki/Amplitude\_damping\_channel) :

```python
>>> K = amplitude_damping_channel(0.2)
```

The variable `K` contains the [Kraus operators](https://en.wikipedia.org/wiki/Quantum\_operation#Kraus\_operators) of the channel. Then,

```python
>>> rho_out = apply_channel(K, rho)
```

gives the state at the output of the channel when the input state is `rho`.

Other functions include:

* Getting the Choi and natural representation of a channel from its Kraus representation
* Converting between the Choi, natural, and Kraus representations of a channel




## Summary of other features

The package also contains functions for:
- Trace norm
- Fidelity and entanglement fidelity
- Random unitaries
- Clifford unitaries
- Generators of the su(d) Lie algebra(for d=2, this is the set of Pauli matrices)
- Discrete Weyl operators
- von Neumann entropy and relative entropy
- Renyi entropies
- Coherent information and Holevo information for states and channels


# Acknowledgements

Thanks to [Mark Wilde](https://www.markwilde.com/) for suggesting the name for the package.



