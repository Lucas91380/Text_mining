import spacy
from datatools import load_dataset
from keras import layers
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

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



#### Mise en forme y ----------------------------------------------- ####

#Binarisation de nos variables à expliquer (les labels)
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(labels)
y_test = label_binarizer.fit_transform(val_labels)

#On retient les valeurs associées aux labels binarisés dans une variable
labelset = label_binarizer.classes_



#### Mise en forme x ----------------------------------------------- ####

#Importation de trucs sur le Français (je ne sais pas encore exactement quoi)
nlp = spacy.load('fr')

#Création du tokenizer
def mytokenize(text):
    doc = nlp(text)
    tokens = [t.text.lower() for sent in doc.sents for t in sent if t.pos_ != "PUNCT" ]
    return tokens

#stopwords
#my_stop_words = text.FRENCH_STOP_WORDS.union(texts)

#Vectorisation (BOW)
vectorizer = CountVectorizer(
    max_features=8000,
    analyzer="word",
    tokenizer=mytokenize,
    stop_words=None,
    ngram_range=(1, 2)
        )

#Vectorisation (tf-idf)
vectorizer = TfidfVectorizer(
    max_features=8000,
    analyzer="word",
    tokenizer=mytokenize,
    stop_words=None,
    ngram_range=(1, 1)
        )

#Vectorisation (tf-idf et n_grames)
vectorizer = TfidfVectorizer(
    max_features=5000,
    analyzer="word",
    #tokenizer=mytokenize,
    stop_words=None,
    ngram_range=(1, 2)
        )


vectorizer.fit(texts)
x_train = vectorizer.transform(texts)
x_test = vectorizer.transform(val_texts)

#### Modèle Keras ----------------------------------------------- ####

#Définition du nombre de neurones dans la couche d'entrée du réseau de neuronnes
input_dim = x_train.shape[1] #Nombres de mots distincts en tout

#Définition du modèle
model = Sequential()
model.add(layers.Dense(units=10, input_dim=input_dim, activation='relu')) #Ajout d'une couche
model.add(layers.Dense(units=3, activation='sigmoid'))

#Configuration du processus d'apprentissage
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Summary de notre modèle (avant de l'entraîner)
model.summary()

#Précision des callbacks (avec un early stopping)
my_callbacks = []
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None)
my_callbacks.append(early_stopping)

#Apprentissage du modèle
history = model.fit(x_train, y_train, epochs=50, verbose=False, callbacks=my_callbacks, validation_data=(x_test, y_test), batch_size=16)

###Qualité du modèle

#Accuracy
loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print("Training accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
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

Plot_history(history)


### Tests de l'accuracy du modèle
items = load_dataset(valfile)

#Valeurs prédites
X = vectorizer.transform(items['text'])
Y = model.predict(X)
slabels = label_binarizer.inverse_transform(Y)

#Valeurs réelles
glabels = items['polarity']

#Calcul de l'accuracy
n = min(len(slabels), len(glabels))
incorrect_count = 0

for i in range(n):
    if slabels[i] != glabels[i]: incorrect_count += 1

acc = (n - incorrect_count) / n
acc = acc * 100
print(acc)