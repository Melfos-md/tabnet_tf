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



--------------------
TODO:
- faire un test de gradient pour sparsemax (comme GLU)
- Ajouter doc dans README.md pour sparsemax