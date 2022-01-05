# Some Collection of Ideas

## Some general thoughts

### Drop the $D$ form the system
This reduces the computational cost of the stages.
But this means that there is no path between input and output that does not contain a state.
This also means that the whole matrix is made up of Hankel matrices.
This in turn means that the rank of these is limited.
This might make it easier to understand it conceptually.

If we walk back, we can understand that including the  $D$ allows the insertion of direct paths form the input to the output that are not constrained by the Hankel Rank.
This might motivate to resort the input and output

### Investigate the ranks of sub matrices
Investigate the rank of submatrices.
This might be done by different means:
- Calculate Ranks for multiple parts. This need thresholding etc.
- Take quadratic matrices and calculate the determinant

## Some ideas on getting Permutations


### Move the largest elements close to the diagonal

Take the largest elements close to the diagonal, such that they can be dealt with by the $D$-matrices.


### Decompose into Sparse+Low rank

The low rank parts can be represented with the Hankel Operators.
The sparse part can be resorted to be as close as possible to the diagonal. This can be represented by the $D_s$.

After this a second optimization step might be possible.

It might also be resorted with consideration to the structure of the low rank parts.
