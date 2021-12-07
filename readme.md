<h1 align="center">CAMNS Method</h1>

The original paper can be found [here](https://www.researchgate.net/profile/Chong-Yung-Chi/publication/251134144_A_Convex_Analysis_Framework_for_Blind_Separation_of_NonNegative_Sources/links/5a1e7622aca272cbfbc04995/A-Convex-Analysis-Framework-for-Blind-Separation-of-NonNegative-Sources.pdf)

![](./img/shuffled.png)
![](./img/extracted.png)

## The problem

The task is blind source separation (BSS) of non-negative source signals. 

Let $N$ - number of sources and $S = [s_1, \ldots, s_N] \in \mathbb{R}^{N \times L}$ - an array of source vectors of length $L$ (flatten images in our case). 
Let $\Omega \in \mathbb{R}^{M \times N}$ - a mixture matrix, where
- $M$ - a number of real observations
- $\forall i \in \{1,\ldots,M\}:\, \sum_{j=1}^{N} \Omega_{i,j} = 1$

So, the observations we have are derived as a product of mixture matrix by sources:
$$
    X = \Omega S
$$

Given:
- $[x_1, \ldots, x_M]$ - array of observations

Want to get:
- $[s_1, \ldots, s_N]$ - sources.

## Assumptions

The CAMNS algorithm works correctly under certain assumptions:

- $\forall i \in \{1,\ldots,N\}:\, s_i \in \mathbb{R}_{+}^{L}$
- Each source vector is _local dominant_: $\forall i \in \{1,\ldots,N\}\,\exists l_i\hookrightarrow s_i[l_i] = 1,\,\text{and}\,\, s_j[l_i] = 0\,(\forall j \neq i)$
- $\forall i \in \{1,\ldots,M\}:\, \sum_{j=1}^{N} \Omega_{i,j} = 1$
- $M \geq N$ and $\Omega$ is of full column rank

## Implementation

- The method is implemented inside [src/camns.py](./src/camns.py) file;
- Also you can see the [demo](./src/demo.ipynb).
