# TabNet_tf

My implementation of TabNet: Attentive Interpretable Tabular Learning  from Sercan O Arik and Tomas Pfister (https://arxiv.org/pdf/1908.07442.pdf) with tensorflow.

Video from Sercan: https://www.youtube.com/watch?v=tQuIcLDO5iE


https://medium.com/deeplearningmadeeasy/sparsemax-from-paper-to-code-351e9b26647b


## Attentive Transformer

![Figure 4d](documentation/images/attentive_transformer.PNG)

- Learnable mask: $\mathbf{M[i]} \in \Re ^ {B \times D}$

- $\mathbf{P[i]} = \prod\nolimits_{j=1}^{i} (\gamma - \mathbf{M[j]})$ where $\gamma$ is a relaxation parameter
- $
\mathbf{M[i]} = \text{sparsemax}(\mathbf{P[i-1]} \cdot \text{h}_i(\mathbf{a[i-1]})).
$
- $\sum\nolimits_{j=1}^{D} \mathbf{M[i]_{b,j}} = 1$

- $L_{sparse} = \sum\nolimits_{i=1}^{N_{steps}} \sum\nolimits_{b=1}^{B} \sum\nolimits_{j=1}^{D} \frac{-\mathbf{M_{b,j}[i]} \log(\mathbf{M_{b,j}[i]} \! +\!  \epsilon)}{N_{steps} \cdot B},$


## Feature processing

From the original paper, the feature transformation is given by:

$[\mathbf{d[i]}, \mathbf{a[i]}] = \text{f}_i(\mathbf{M[i]} \odot \mathbf{f})$

where:
- $B$ denotes the batch size.
- $D$ represents the number of features.
- $\mathbf{f} \in \Re ^ {B \times D}$ is the matrix of input features.
- $\mathbf{M[i]} \in \Re ^ {B \times D}$ is the learnable mask applied to the features.
- $\mathbf{d[i]} \in \Re ^ {B \times N_d}$ and $\mathbf{a[i]} \in \Re ^ {B \times N_a}$ are the outputs of the transformation.

Following this:
- $\mathbf{M[i]} \odot \mathbf{f}$ is an element-wise multiplication so the result is of shape $(B, D)$
- Consequently, $[\mathbf{d[i]}, \mathbf{a[i]}]$ possesses a shape of $(B, N_a + N_d)$

The function $\text{f}_i$ is the learnable transformation which includes fully connected (FC) layers. Each FC layer should have $N_a + N_d$ neurons.The resulting matrix is then split with the first $N_d$ rows directed to $d[i]$ and the remaining $N_a$ rows directed to $a[i]$.


***Note on the $\sqrt{0.5}$***


Residual connections in neural networks involve adding the output of one layer to the output of one or more previous layers. This can be beneficial in deep architectures to mitigate the vanishing gradient problem and improve convergence.

When performing an element-wise addition of two tensors, the variance of the output can increase. To understand why, consider:

Let $X$ and $Y$ be two independent random variables with the same variance $\sigma^2$. The variance of their sum is:

$\text{Var}(X + Y) = 2 \times \sigma^2$

To keep the variance of the sum consistent with $\sigma^2$, we would divide the sum by $\sqrt{2}$:

$\text{Var} \left( \frac{X + Y}{\sqrt{2}} \right) = \sigma^2$

This reasoning underpins the division by $\sqrt{2}$ (= multiply by $\sqrt{0.5}$) after adding residual connections. It aims to stabilize variance across layers.

Reference: Gehring, J.; Auli, M.; Grangier, D.; Yarats, D.; and Dauphin,
Y. N. 2017. Convolutional Sequence to Sequence Learning.
arXiv:1705.03122 .

## Utils

### Gated Linear Unit

**Reference**: Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). Language Modeling with Gated Convolutional Networks. Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML'17, 933–941. ([arXiv](https://arxiv.org/pdf/1612.08083.pdf)).

The Gated Linear Unit (GLU) activation function is employed in TabNet. The authors observed an empirical advantage of using GLU over conventional nonlinearities like ReLU.

What I believe: GLUs offer a more refined modulation of information flowing through the network, allowing for a nuanced blend of features, in contrast to the more binary behavior of ReLU (either activated or not). This nuanced control might provide TabNet with enhanced flexibility and performance in handling diverse feature interactions


$\text{GLU}(x) = x \odot \sigma(Wx + b)$

where:
- $x$ is the input
- $\sigma$ is the sigmoid function
- $W$ is the weight matrix
- $b$ the bias vector

In TabNet, the linear transformation is handled by the FC layer. Thus, my implementation of GLU will focus on the sigmoid activation and the element-wise multiplication.


### Ghost BatchNormalization


**Reference**: Neofytos Dimitriou, Ognjen Arandjelovic (2020). A New Look at Ghost Normalization. ([arXiv](https://arxiv.org/pdf/2007.08554.pdf)).


Ghost Batch Normalization (GBN) is based on the idea of Batch Normalization (BN), but with a slight twist. In traditional BN, statistics (mean and variance) are computed over the entire batch. In GBN, these statistics are computed over "ghost batches", which are essentially subsets of the original batch

Suppose you have a full batch $B$ splitted into $k$ ghost batches $B_1, B_2, ..., B_k$.

For a specific ghost batch $B_j$, the normalization formula goes as follows:

- Mean

$\mu_{B_j} = \frac{1}{m} \sum_{i=1}^{m} x_i$

where $m$ is the size of the ghost batch $B_j$ and $x_i$ are the inputs within $B_j$.

- Variance
  
$\sigma_{B_j}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{B_j})^2$

- Normalization

$\hat{x_i} = \frac{x_i - \mu_{B_j}}{\sqrt{\sigma_{B_j}^2 + \epsilon}}$

where $\epsilon$ is a small number to prevent division by zero.

- Scaling and shift

$y_i = \gamma \hat{x_i} + \beta$


where $\gamma$ and $\beta$ are learnable parameters.

This process is repeated for each ghost batch $B_j$. So, rather than using statistics over the entire full batch, you use the statistics from the relevant ghost batch for each input.

--------------------
TODO:
- faire un test de gradient pour sparsemax (comme GLU)
- Ajouter doc dans README.md pour sparsemax
- Implementer L_sparse