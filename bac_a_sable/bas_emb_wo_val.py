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
maxlen = 100
x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)

#Binarisation de nos variables à expliquer (les labels)
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(labels)
y_test = label_binarizer.fit_transform(val_labels)

#On retient les valeurs associées aux labels binarisés dans une variable
labelset = label_binarizer.classes_


#### Modèle Keras ----------------------------------------------- ####

####Tuto
from keras import layers
from keras.models import Sequential
from keras.callbacks import EarlyStopping

#Taille de la sortie de la couche d'embedding
embedding_dim = 50

#Définition du modèle
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(units=10, activation='relu'))
model.add(layers.Dense(units=3, activation='sigmoid'))

#Configuration du processus d'apprentissage
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Summary de notre modèle (avant de l'entraîner)
model.summary()

#Précision des callbacks (avec un early stopping)
my_callbacks = []
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None)
my_callbacks.append(early_stopping)

#Apprentissage du modèle
history = model.fit(x_train, y_train, epochs=200, verbose=False, callbacks=my_callbacks, batch_size=16)


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