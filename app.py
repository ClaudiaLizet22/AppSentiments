#--------------------------------------------------------------------------------------------
#  Copyright (c) Red Hat, Inc. All rights reserved.
#  Licensed under the MIT License. See LICENSE in the project root for license information.
#--------------------------------------------------------------------------------------------

# This program prints Hello, world!

#Local imports
import datetime
import pickle5 as pickle
#Third part imports
import json
from flask import Flask, request, jsonify
import pandas as pd


model_name= "Sentiment analisys"
model_file='model.model.pkl'  
version= 'v1.0.0'
app=Flask(__name__)
model = None
vectorizador = None
encoder = None
  

import contractions
from nltk.tokenize import TweetTokenizer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp
import spacy
import nltk
from sklearn.preprocessing import LabelEncoder
#run local
#cp -r nltk_data/ /home/user/
#export NKTL_DATA=/projects/sentimentanalysis/nltk_data/
#nltk.download('stopwords',download_dir='/projects/sentimentanalysis/nltk_data/')
nltk.download('stopwords',download_dir='nltk_data')

with open ('model.pkl','rb') as archivo :
    model=pickle.load(archivo)

with open ('vectorizador.pkl', 'rb') as fvectorizer:
    vectorizador = pickle.load(fvectorizer)

with open ('encoder.pkl', 'rb') as fvectorizer:
    encoder = pickle.load(fvectorizer)  

def tokenize(texto):
  tweet_tokenizer = TweetTokenizer()
  tokens_list = tweet_tokenizer.tokenize(texto)
  return tokens_list

# Quitar stop words de una lista de tokens
def quitar_stopwords(tokens):
    stop_words = nltk.corpus.stopwords.words('english')
    filtered_sentence = [w for w in tokens if not w in stop_words]
    return filtered_sentence


# Eliminar signos de puntuación de una lista de tokens
# (nos quedamos sólo lo alfanumérico en este caso)
def quitar_puntuacion(tokens):
    words=[word for word in tokens if word.isalnum()]
    return words


# Lemmatization de los tokens. Devuelve una string entera para hacer la tokenización
# con NLTK
en_core_web_sm = spacy.load('en_core_web_sm')
nlp = en_core_web_sm
def lematizar(tokens):
    sentence = " ".join(tokens)
    mytokens = nlp(sentence)
    # Lematizamos los tokens y los convertimos  a minusculas
    mytokens = [ word.lemma_ if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    # Extraemos el text en una string
    return " ".join(mytokens)

def replace_contractions(text):
    expanded_words=[]
    for t in text.split():
        expanded_words.append(contractions.fix(t,slang=True))
    expanded_text = ' '.join(expanded_words)
    return expanded_text

def get_model_response(json_data):
    #transformar
    new_text = json_data['Review'] + ' ' + json_data['Summary']
    summary_processed = replace_contractions(new_text)
    tokenized = tokenize(summary_processed)
    tokenized_clean = quitar_stopwords(tokenized)
    lematizacion = lematizar(tokenized_clean)
    newsentiment = TextBlob(lematizacion).sentiment.polarity
    vector_data_new = vectorizador.transform([lematizacion])
    extra_features = [newsentiment]
    X_new = sp.sparse.hstack((vector_data_new,extra_features),format='csr')

    #fin transformar
    prediction = model.predict(X_new)
    result = encoder.inverse_transform(prediction)
    return {
        'status': 200,
        'prediction': result[0]
    }


@app.route('/info', methods=["GET"])
def info():
    """Return model information, version, how to call"""
    result={}

    result["name"] = model_name
    result["version"]=version

    return jsonify(result)

@app.route('/healt',methods=["GET"])
def healt():
    """return service healt"""
    return 'ok'

@app.route('/predict',methods=['POST'])
def predict():
    feature_dict= request.get_json()
    if not feature_dict:
        return{
            'error':'Body is empty' 
        },500
    try:
        response= get_model_response(feature_dict)
    except ValueError as e:
        return{'error': str(e).split('/n')[-1].strip()},500
   
    return response,200


if __name__=='__main__':
    app.run(host='0.0.0.0') # direccion de la maquina donde se ejecuta el proceso en mi mismo equipo -el servidor corre en local
