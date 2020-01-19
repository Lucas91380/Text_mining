#### Importation ----------------------------------------------- ####

#### Exam
from datatools import load_dataset
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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
maxlen = 1000
x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)

#Binarisation de nos variables à expliquer (les labels)
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(labels)
y_test = label_binarizer.fit_transform(val_labels)

#On retient les valeurs associées aux labels binarisés dans une variable
labelset = label_binarizer.classes_

#### Embedding français ----------------------------------------------- ####
import numpy as np
from gensim.models import KeyedVectors as kv
emb_filename = "../resources/frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin"
emb = kv.load_word2vec_format(emb_filename, binary=True, encoding='UTF-8', unicode_errors='ignore')
emb_dim = emb.vector_size
dir(emb)

#Word embedding préentrainé
padid = 0
padVector = np.zeros(emb_dim)
emb_weights = emb.vectors
emb_weights = np.insert(emb_weights, padid, padVector, axis=0)

#Fonction pour créer notre embedding matrix
def Select_words(emb, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word in emb.vocab:
        if word in word_index:
            idx = word_index[word]
            embedding_matrix[idx] = np.array(emb.word_vec(word), dtype=np.float32)[:embedding_dim]

    return embedding_matrix

#Création de la matrice
embedding_dim = 200
embedding_matrix = Select_words(emb, tokenizer.word_index, embedding_dim)

#Proportion du vocabuaire couvert par le modèle préentrainé
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
nonzero_elements / vocab_size

#### Modèle Keras ----------------------------------------------- ####

####Tuto
from keras import layers
from keras.models import Sequential
from keras.callbacks import EarlyStopping

#Définition du modèle
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(units=10, activation='relu'))
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
history = model.fit(x_train, y_train, epochs=20, verbose=False, callbacks=my_callbacks, validation_data=(x_test, y_test), batch_size=16)

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
X = tokenizer.texts_to_sequences(items['text'])
X = pad_sequences(X, padding='post', maxlen=maxlen)
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