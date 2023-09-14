# TabNet_tf

My implementation of TabNet: Attentive Interpretable Tabular Learning  from Sercan O Arik and Tomas Pfister (https://arxiv.org/pdf/1908.07442.pdf) with tensorflow.

Video from Sercan: https://www.youtube.com/watch?v=tQuIcLDO5iE


https://medium.com/deeplearningmadeeasy/sparsemax-from-paper-to-code-351e9b26647b


## Attentive Transformer

![Figure 4d](documentation/images/attentive_transformer.PNG)


The output shape of this layer is (1, num_features).
While the original paper state that the mask shape is (batch_size, num_features) ,
I chose to apply the same mask for one step to all training examples.

"L'intérêt serait plutôt d'avoir le même masque pour tous les exemples d'entrainement, et ce masque change à chaque étape i de l'algorithme, chaque étape i représentant une attention particulière. Le fait fait de créer un masque avec des différences entre les exemples d'entrainement va rendre l'algorithme sensible à l'ordre dans lequel sont saisie les exemples d'entrainement ou bien la façon dont ils sont organisés (ordre alphabétique d'une caractéristique par exemple) et c'est quelque chose qui n'apporte pas d'information pertinente à mon avis"

Bon vu la complexité, je vais rester sur l'article original. Je pousserai la réflexion dans un second temps en créant ma propre approche.


