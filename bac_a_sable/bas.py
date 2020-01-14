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