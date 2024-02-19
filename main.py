from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
from nltk.corpus import stopwords
import pandas as pd
from fastapi import FastAPI
import spacy
import numpy as np
app = FastAPI()

def clean_stopWords(rows):
    stopwords_spanish = stopwords.words('spanish')
    no_stops = []
    for word in rows:
        if word not in stopwords_spanish:
            no_stops.append(word)
    return no_stops


def lematizer_text(rows):
    lemmer = spacy.load('es_core_news_sm')
    doc = [word.lemma_ for word in lemmer(rows)]

    return doc

def alf_nums(rows):
    cadena = []
    for word in rows:
        if (word is not word.isdigit()) or (word is not word.isalnum()):
            try:
                float(word)
            except ValueError:
                cadena.append(word)

    return cadena


text = pd.read_parquet('data_cleaned.parquet')['review_text']

text = text.map(lambda corpus: ' '.join(corpus))
text_vectorizer = TextVectorization(output_mode='int')

text_vectorizer.adapt([text])

model = Sequential(name='Text_Vectorizing')
model.add(Input(shape=(1,), dtype=tf.string))
model.add(text_vectorizer)

def text_cleaner(row):

    lematized = lematizer_text(row)
    no_stops = clean_stopWords(lematized)


    text_vectorized = model.predict([' '.join(no_stops)])

    return text_vectorized


modelo = tf.keras.models.load_model('RNN_model.h5')

@app.get("/Comentario/")
def text(comentario : str):
    padd = sequence.pad_sequences(text_cleaner(comentario), maxlen=100, padding='post', truncating='post')

    rest = modelo.predict(padd.reshape(1,100,1))
    
    if rest[0][0]> rest[0][1]:
        return ['Comentario Negativo']
    else:
        return ['Comentario Positivo']

        

