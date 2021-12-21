Quasiderivative 1-Norm

Quasiderivative 1-Norm

$$g(x) = \|x\|_1$$

with the definition

$$g(x) = \sum_{i=1}^n |x_i|$$

This can be rewritten as 

$$g(x) = \sum_{i=1}^n |x_i| = \sum_{i=1}^n \operatorname{sgn}(x_i) x_i$$

We move on to the derivative

$$\mathrm{D}_{V}g(x) = \frac{\mathrm d}{\mathrm d t}\bigg|_{t=0}  \sum_i \operatorname{sgn}\big([x+tv]_{i}\big) [x+tv]_{i}$$

$$=   \sum_i \frac{\mathrm d}{\mathrm d t}\bigg|_{t=0} \operatorname{sgn}\big([x+tv]_{i}\big) [x+tv]_{i}$$

$$=   \sum_i  \operatorname{sgn}\big([x]_{i}\big) [v]_{i}$$

To get the derivative we have to write it as $D_vg(x)$ as $v^\top a$

This is 
$$D_vg(x)= v^\top \operatorname{sgn}(g) $$

Therefore the (Quasi) Derivative is $\operatorname{sgn}(g)$


## But in this case we have a $x = Aa$

$$g(a) = \|Aa\|_1$$

Using the Formula for Matrix-vector multiplications

$$g(a) = \sum_{i=1}^n |x_i| = \sum_{i=1}^n \left|\sum_{j=1}^n A_{ij} a_j \right| $$

$$\mathrm{D}_{V}g(a) = \frac{\mathrm d}{\mathrm d t}\bigg|_{t=0}  \sum_i \operatorname{sgn}\left(\sum_{j=1}^n A_{ij} [a+tv]_j \right) \left(\sum_{j=1}^n A_{ij} [a+tv]_j \right)$$

$$=  \sum_i \frac{\mathrm d}{\mathrm d t}\bigg|_{t=0} \operatorname{sgn}\left(\sum_{j=1}^n A_{ij} [a+tv]_j \right) \left(\sum_{j=1}^n A_{ij} [a+tv]_j \right)$$

with $\operatorname{sgn}(x) = const$ and product rule

$$=  \sum_i \operatorname{sgn}\left(\sum_{j=1}^n A_{ij} [a]_j \right) \frac{\mathrm d}{\mathrm d t}\bigg|_{t=0} \left(\sum_{j=1}^n A_{ij} [a]_j +\sum_{j=1}^n A_{ij} [tv]_j \right)$$

$$=  \sum_i \operatorname{sgn}\left(\sum_{j=1}^n A_{ij} [a]_j \right)  \left(\sum_{j=1}^n A_{ij} [v]_j \right)$$

$$=   \sum_i  \operatorname{sgn}\big([Aa]_{i}\big) [Av]_{i}$$

$$=   \operatorname{sgn}\big(Aa\big)^\top Av = v^\top \left(\operatorname{sgn}\big(Aa\big)^\top A \right)^\top = v^\top A^\top \operatorname{sgn}(Aa)$$

This geives the Quasi-derivative 

## Sum of multiple L1-Norms 

From a MORL exercice

As $g(\cdot)$ we use the $l_1$-norm
$$g(A)= \sum_i \sum_j |(A)_{ij}|$$

The function is not diff-able for $(A)_{ij}=0$ but we can calculate a
derivative for other points. We rewrite the absolute value $|x|$ as
$\operatorname{sgn}(x)x$. Where $\operatorname{sgn}$ is the sign
function. (Which is also not diff-able for $x=0$)

This gives

$$g(\Omega X)= \sum_i \sum_j \operatorname{sgn}\big((\Omega X)_{ij}\big) (\Omega X)_{ij}$$

Based on this we calculate the derivatives with respect to $\Omega$.
The sign function will be treated as constant.

$$\mathrm{D}_{V}g(\Omega X) = \frac{\mathrm d}{\mathrm d t}\bigg|_{t=0} \sum_i \sum_j \operatorname{sgn}\big(((\Omega+tV)X)_{ij}\big) ((\Omega+tV)X)_{ij}$$

$$=\sum_i \sum_j \frac{\mathrm d}{\mathrm d t} \bigg|_{t=0} \operatorname{sgn}\big(((\Omega+tV)X)_{ij}\big) (\Omega X+tVX)_{ij}$$

$$=\sum_i \sum_j  \operatorname{sgn}\big((\Omega X)_{ij}\big) (VX)_{ij}$$

  To get the gradient the eqation has too be brought io the form
$\mathrm{tr}(V^\top A)$

For this we can construct a matrix that has
$$\sum_j  \operatorname{sgn}\big((\Omega X)_{ij}\big) (VX)_{ij}$$

on the diagonal.

We can use the domula for an Matrix-Matrix multiplication

$$(C)_{ik}=\sum_{j=1}^m (A)_{ij} (B)_{jk}$$

Here only the diagional elemnts with $k=i$ are interessting:

$$(C)_{ii}=\sum_{j=1}^m (A)_{ij} (B)_{ji}$$

By transposing the decond matrix we get

$$(C)_{ii}=\sum_{g=1}^m (A)_{ij} (B^\top)_{ji}$$

This has the desired form with $A = \mathrm{sgn}(\Omega X)$ and
$B^\top=VX$

This results in
$$\mathrm{D}_{V}g(\Omega X)= \mathrm{tr}\big(\mathrm{sgn}(\Omega X) X^\top V^\top\big)=\mathrm{tr}\big(V^\top\mathrm{sgn}(\Omega X) X^\top \big)$$

and the gradient

$$\nabla g(\Omega X) = \mathrm{sgn}(\Omega X) X^\top$$

If we use a implementation with $\mathrm{sgn}(0)=0$ we obtian this as
a derivative. This does not lead to negative effects.
