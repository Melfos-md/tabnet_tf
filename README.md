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

TODO:
- Documentation : Bien que vous ayez fourni une docstring, il serait bon d'inclure également des commentaires tout au long du code pour expliquer chaque étape, surtout dans des parties plus complexes comme la mise à jour de prior_scales.

- Erreurs potentielles : Dans la méthode build, vous vérifiez si len(input_shape) != 2. Cependant, si input_shape est (batch_size, num_features), alors len(input_shape) sera toujours 2, indépendamment de la valeur de batch_size ou num_features. Vous pourriez plutôt vouloir vérifier if len(input_shape) != 2 or input_shape[0] is None: pour vous assurer que la forme d'entrée est correcte.

- parsemax : Vous avez utilisé Sparsemax comme fonction d'activation. Assurez-vous d'avoir la bonne implémentation de Sparsemax et d'avoir les dépendances nécessaires pour cela.

- Initialisation : Vous utilisez GlorotUniform pour l'initialisation, ce qui est bien. Cependant, vous pouvez envisager d'expérimenter avec d'autres initialisateurs pour voir s'il y a une différence de performance.

- Fonctions d'activation : Vous utilisez une activation None pour votre couche dense. Selon le papier original de TabNet, une activation de type ReLU peut également être utilisée. Vous pourriez envisager d'ajouter cela en tant que paramètre pour permettre une certaine flexibilité.