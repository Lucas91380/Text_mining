#### Importation ----------------------------------------------- ####

#### Tuto
import pandas as pd

#Dictionnaire des chemins
filepath_dict = {
        'yelp' : 'data/sentiment_analysis/yelp_labelled.txt',
        'amazon' : 'data/sentiment_analysis/amazon_cells_labelled.txt',
        'imbd' : 'data/sentiment_analysis/imdb_labelled.txt'
        }

#Importation des trois fichiers
df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source #Ajoute une colonne avec le nom de la source (imbd, amazon ou yelp)
    df_list.append(df)

#Concaténation des 3 sources en une seule dataframe
df_tuto = pd.concat(df_list)


#### Exam
from datatools import load_dataset

#Chemin des données
datadir = "../data/"
trainfile =  datadir + "frdataset1_train.csv"

#Importation des données
df_exam = load_dataset(trainfile)

#texts et labels ne sont que les variables de df_exam qu'on a isolées
texts = df_exam['text']
labels = df_exam['polarity']



#### Modèle de référence ----------------------------------------------- ####

#### Tuto
from sklearn.model_selection import train_test_split

#On ne prend que les données venant de yelp
df_yelp = df_tuto[df_tuto['source'] == 'yelp']

#On isole les commentaires et leur valeur associée et les mettons au format NumPy (plus commode)
texts_yelp = df_yelp['sentence'].values
labels_yelp = df_yelp['label'].values

#Séparation en train test
texts_yelp_train, texts_yelp_test, labels_yelp_train, labels_yelp_test = train_test_split(texts_yelp, labels_yelp, test_size=0.25, random_state=1000)

#Vectorisation (BOW)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_tuto = CountVectorizer()
vectorizer_tuto.fit(texts_yelp_train)
x_train_tuto = vectorizer_tuto.transform(texts_yelp_train)
x_test_tuto = vectorizer_tuto.transform(texts_yelp_test)

#Régression logistique pour modèle de référence
from sklearn.linear_model import LogisticRegression
classifier_tuto = LogisticRegression()
classifier_tuto.fit(x_train_tuto, labels_yelp_train)

#Tx de bonnes prédictions de notre modèle de référence
classifier_tuto.score(x_test_tuto, labels_yelp_test)


#### Premier modèle Keras ----------------------------------------------- ####

####Tuto
from keras.models import Sequential
from keras import layers

#Définition du nombre de neurones dans la couche d'entrée du réseau de neuronnes
input_dim_tuto = x_train_tuto.shape[1] #Nombres de mots distincts en tout

#Définition du modèle
model_tuto = Sequential()
model_tuto.add(layers.Dense(units=10, input_dim=input_dim_tuto, activation='relu')) #Ajout d'une couche
model_tuto.add(layers.Dense(units=1, activation='sigmoid'))

#Configuration du processus d'apprentissage
model_tuto.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Summary de notre modèle (avant de l'entraîner)
model_tuto.summary()

#Apprentissage du modèle
history_tuto = model_tuto.fit(x_train_tuto, labels_yelp_train, epochs=100, verbose=False, validation_data=(x_test_tuto, labels_yelp_test), batch_size=10)

###Qualité du modèle

#Accuracy
loss, accuracy = model_tuto.evaluate(x_train_tuto, labels_yelp_train, verbose=False)
print("Training accuracy: {:.4f}".format(accuracy))
loss, accuracy = model_tuto.evaluate(x_test_tuto, labels_yelp_test, verbose=False)
print("Testing accuracy: {:.4f}".format(accuracy))

#P'tit graphique sympa pour regarder l'évolution de l'accuracy et de la fonction de perte
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def Plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc)+1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

Plot_history(history_tuto)
