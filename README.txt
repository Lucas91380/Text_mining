Eudiants :  Etienne Hamard (M2 SSD) et Lucas Chabeau (M2 SSD)

L'objectif était ici de créer un modèle qui prédit la "positivité" d'un message parmis trois catégories : {positive, neutral, negative}.
Nous avons testé plein de méthodes différentes avant d'arriver à celle retenue. Nous avons tenté des vectoriser nos données explicatives
en bag of words, embeddings et tf-idf. C'est finalement la représentation tf-idf des commentaires (avec n-grammes allant de 1 à 4) qui 
s'est avérée la plus concluante. Nous la limitons à 27000 éléments par observation et obtenons notre matrice explicative que nous passons
dans un réseau neuronnes (perceptron multiple) avec les couches suivantes :
	- une couche cachée intermédiaire qui sort 128 noeuds à l'aide de la fonction d'activation d'unité de rectification linéaire ('relu') 
	- une couche de sortie de taille 3 avec la fonction d'activation softmax
Ce réseau de neuronne utilise l'enthropie croisée catégorielle (categorial crossentropy) comme fonction de perte et est résolu par
l'optimiseur adam.
Le nombre d'itérations de notre réseau de neuronnes (epochs) est limité à 50, mais nous n'observons jamais ce cas, l'execution
s'arrête dès que la fonction de perte devient croissante (deux itérations de latence). Cet évènement arrive toujours bien avant
les 50 itérations dans notre cas. Pour la partie d'apprentissage du modèle, nous découpé notre matrice d'entrée en 64 "batchs".
Notre classifieur n'utilise finalement pas d'autre ressources extérieures que le modèle français de spacy.

Nous obtenons avec ce classifieur une moyenne de 66.36% de bonnes présdictions sur les données dev.