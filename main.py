from tensorflow.keras.preprocessing import sequence
from nltk.corpus import stopwords

stopwords_spanish = stopwords.words('spanish')
def clean_stopWords(rows):
    no_stops = []
    for word in rows:
        if word not in stopwords_spanish:
            no_stops.append(word)
    return no_stops


if __name__=='__main__':
    print('')
    
