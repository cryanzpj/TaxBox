import pickle
import pandas as pd
from chatbot import load_tf_model,response

def Bin_loader():
    tfidf_model_filename = './Bin/tfidf_vectorizer.sav'
    tfidf_filename = './Bin/tfidf.sav'

    loaded_tfidf_vectorizer = pickle.load(open(tfidf_model_filename, 'rb'))
    loaded_tfidf = pickle.load(open(tfidf_filename, 'rb'))

    data_filename = './Bin/taxcode_data_sentences_subchapterD.csv'

    colnames = ['id', 'sid', 'mid', 'title', 'text']
    df_data = pd.read_csv(data_filename, names=colnames, header=None, encoding='utf-8')

    inten_model,  intent_words, intent_classes,intents = load_tf_model()

    return {'loaded_tfidf': loaded_tfidf, "loaded_tfidf_vectorizer": loaded_tfidf_vectorizer, "df_data": df_data,
            'inten_model':inten_model, 'intent_words':intent_words,"intent_classes":intent_classes,"intents":intents}


