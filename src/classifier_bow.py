import numpy as np
np.random.seed(15)

import spacy
from datatools import load_dataset
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras import layers
from keras.models import Sequential
from keras.callbacks import EarlyStopping


nlp = spacy.load('fr')


class Classifier:
    """The Classifier"""

    def __init__(self):
        self.labelset = None
        self.label_binarizer = LabelBinarizer()
        self.model = None
        self.epochs = 50
        self.batchsize = 64
        self.max_features = 27000
        # create the vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            strip_accents=None,
            analyzer="word",
            tokenizer=self.mytokenize,
            stop_words=None,
            ngram_range=(1, 4),
            binary=False,
            preprocessor=None
        )

    def mytokenize(self, text):
        """Customized tokenizer.
        Here you can add other linguistic processing and generate more normalized features
        """
        doc = nlp(text)
        tokens = [t.text.lower() for sent in doc.sents for t in sent if t.pos_ != "PUNCT" ]
        # tokens = [t for t in tokens if t not in self.stopset]
        return tokens

    def feature_count(self):
        return len(self.vectorizer.vocabulary_)

    def create_model(self):
        """Create a neural network model and return it.
        Here you can modify the architecture of the model (network type, number of layers, number of neurones)
        and its parameters"""

        #Définition du nombre de neurones dans la couche d'entrée du réseau de neuronnes
        input_dim = self.feature_count() #Nombres de mots distincts en tout

        #Définition du modèle
        model = Sequential()
        model.add(layers.Dense(units=128, input_dim=input_dim, activation='relu')) #Ajout d'une couche
        model.add(layers.Dense(units=3, activation='softmax'))

        #Configuration du processus d'apprentissage
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #Summary de notre modèle (avant de l'entraîner)
        model.summary()

        return model




    def vectorize(self, texts):
        return self.vectorizer.transform(texts)


    def train_on_data(self, texts, labels, valtexts=None, vallabels=None):
        """Train the model using the list of text examples together with their true (correct) labels"""
        # create the binary output vectors from the correct labels
        Y_train = self.label_binarizer.fit_transform(labels)
        # get the set of labels
        self.labelset = set(self.label_binarizer.classes_)
        print("LABELS: %s" % self.labelset)
        # build the feature index (unigram of words, bi-grams etc.)  using the training data
        self.vectorizer.fit(texts)
        # create a model to train
        self.model = self.create_model()
        # for each text example, build its vector representation
        X_train = self.vectorize(texts)
        #
        my_callbacks = []
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto', baseline=None)
        my_callbacks.append(early_stopping)
        if valtexts is not None and vallabels is not None:
            X_val = self.vectorize(valtexts)
            Y_val = self.label_binarizer.transform(vallabels)
            valdata = (X_val, Y_val)
        else:
            valdata = None
        # Train the model!


        self.model.fit(
            X_train, Y_train,
            epochs=self.epochs,
            batch_size=self.batchsize,
            callbacks=my_callbacks,
            validation_data=valdata,
            verbose=2)

    def predict_on_X(self, X):
        return self.model.predict(X)


    def predict_on_data(self, texts):
        """Use this classifier model to predict class labels for a list of input texts.
        Returns the list of predicted labels
        """
        X = self.vectorize(texts)
        # get the predicted output vectors: each vector will contain a probability for each class label
        Y = self.model.predict(X)
        # from the output probability vectors, get the labels that got the best probability scores
        return self.label_binarizer.inverse_transform(Y)


    ####################################################################################################
    # IMPORTANT: ne pas changer le nom et les paramètres des deux méthode suivantes: train et predict
    ###################################################################################################
    def train(self, trainfile, valfile=None):
        df = load_dataset(trainfile)
        texts = df['text']
        labels = df['polarity']
        if valfile:
            valdf = load_dataset(valfile)
            valtexts = valdf['text']
            vallabels = valdf['polarity']
        else:
            valtexts = vallabels = None
        self.train_on_data(texts, labels, valtexts, vallabels)


    def predict(self, datafile):
        """Use this classifier model to predict class labels for a list of input texts.
        Returns the list of predicted labels
        """
        items = load_dataset(datafile)
        return self.predict_on_data(items['text'])




