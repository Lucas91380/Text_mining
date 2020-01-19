from datatools import load_dataset
from sklearn.preprocessing import LabelBinarizer
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV


#### Importation ----------------------------------------------- ####

#Chemin des données
datadir = "../data/"
trainfile =  datadir + "frdataset1_train.csv"
valfile = datadir + "frdataset1_dev.csv"

#Importation des données
df_train = load_dataset(trainfile)
df_val = load_dataset(valfile)

#texts et labels ne sont que les variables de df_train qu'on a isolées
texts = df_train['text'].values
labels = df_train['polarity'].values
val_texts = df_val['text'].values
val_labels = df_val['polarity'].values

#Tokenisation de nos données explicatives
tokenizer = Tokenizer(num_words=8000)
tokenizer.fit_on_texts(texts)
x_train = tokenizer.texts_to_sequences(texts)
x_test = tokenizer.texts_to_sequences(val_texts)

#Taille du vocabulaire
vocab_size = len(tokenizer.word_index) + 1

#On remplit de 0 les vecteurs tokenizés pour chaque phrase avoir des vecteurs de tailles identiques
maxlen = 100
x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)

#Binarisation de nos variables à expliquer (les labels)
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(labels)
y_test = label_binarizer.fit_transform(val_labels)

#On retient les valeurs associées aux labels binarisés dans une variable
labelset = label_binarizer.classes_

#### Optimisation ----------------------------------------------- ####

#Principaux paramètres
epochs = 20
embedding_dim = 50
output_file = 'optim.txt'

#Fonction qui créé le modèle
def Crea_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(3, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


#Grille de paramètres
param_grid = dict(
        num_filters = [32, 64, 128],
        kernel_size = [3, 5, 7],
        vocab_size = [vocab_size],
        embedding_dim = [embedding_dim],
        maxlen = [maxlen]
        )

#Création du modèle
model = KerasClassifier(
        build_fn=Crea_model,
        epochs=epochs,
        batch_size=16,
        verbose=False
        )

#Création de la grille (aléatoire, pas grid search)
grid = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_grid,
        cv = 5,
        verbose = 1,
        n_iter = 1
        )

#Execution de la grille
grid_result = grid.fit(x_train, y_train)

#Evaluation des résultats
test_accuracy = grid.score(x_test, y_test)

#Affichage des résultats de la grille
print(grid_result.best_score_, grid_result.best_params_, test_accuracy)