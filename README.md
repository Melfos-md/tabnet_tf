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
- FeatureTransformer: Comme vous l'avez mentionné, il y a un bloc partagé et un bloc indépendant pour chaque étape.

- FeatureTransformerShared: Une classe pour le bloc partagé qui sera utilisé dans chaque étape.
FeatureTransformerStep: Une classe pour le bloc indépendant spécifique à chaque étape.

- TabNetBlock: Une classe qui combine le FeatureTransformer (blocs partagé et indépendant) et l'AttentiveTransformer. Ceci est essentiellement une étape complète de TabNet.

Cette classe va orchestrer:

Le passage des données à travers le FeatureTransformer partagé.
Le passage des données à travers le FeatureTransformer pour une étape spécifique.
L'utilisation de l'AttentiveTransformer pour générer le masque.
L'application du masque à la sortie du FeatureTransformer.

- TabNetModel: Une classe pour l'ensemble du modèle TabNet qui contient N_steps instances du TabNetBlock (où N_steps est le nombre d'étapes).

Cette classe sera responsable de :

Initialiser les différentes étapes.
Orchestrer le flux de données à travers chaque étape.
Calculer la perte, y compris le terme de régularisation L_sparse.
Toutes les autres responsabilités liées au modèle, telles que la sauvegarde/chargement des poids, etc.

- Utils: Diverses fonctions et classes utilitaires.

Par exemple:

SparseRegularization: Si vous voulez modulariser la fonction de perte pour L_sparse.
Sparsemax: Comme vous l'avez déjà fait.
Toute autre fonctionnalité d'aide ou couche personnalisée que vous pourriez nécessiter.
- Main / Trainer: Une classe ou un script pour entraîner, évaluer, et tester votre modèle. Elle gérera la boucle d'entraînement, la validation, l'enregistrement des checkpoints, etc.