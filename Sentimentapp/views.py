from django.shortcuts import render

from django.shortcuts import render
from django.http import HttpResponse
from keras.models import load_model
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pandas as pd
import numpy as np
import joblib


token=joblib.load(r"C:\Users\PMLS\Desktop\New folder (9)\Sanaylysisproject\token4.pkl")
ps = PorterStemmer()

model=load_model(r"C:\Users\PMLS\Desktop\New folder (9)\Sanaylysisproject\sentimentanysis95acc.h5")


# Get max sequence length
max_sequence_length = 26


def home(request):
    # if request.method == 'POST':
    #     text = request.POST.get('text')
    #     method = request.POST.get('method')
    #     if method == 'ai_model':
    #         sentiment, emoji = analyze_sentiment(text)  # Call sentiment analysis function
    #         return render(request, 'index.html', {'sentiment': sentiment, 'emoji': emoji})
    #     # Add more methods if needed
    return render(request, 'nfile.html')



def gsanaylysis(request):
    if request.method == 'POST':
        # Handle form submission
        text = request.POST.get('text_input')
        text = text.lower()
        print(text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\W', ' ', text)
        tokens = word_tokenize(text)
        print(tokens)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        print(tokens)
        stems = [ps.stem(word) for word in tokens]
        preprocessed_text = ' '.join(stems)
        print(preprocessed_text)

        # Convert the preprocessed text to a sequence
        sequence = token.texts_to_sequences([preprocessed_text])
        print(sequence)
        sequence = pad_sequences(sequence,padding="pre", maxlen=26)

        # Predict the sentiment
        prediction = model.predict(sequence)
        print(np.argmax(prediction))

        if np.argmax(prediction) == 0:
            sentiment = "Negative"
        elif np.argmax(prediction) == 1:
            sentiment = "Negative"
        elif np.argmax(prediction) == 2:
            sentiment = "Neutral"
        else:
            sentiment = "Positive"
        
       
        #    result=predict_sentiment(dropdown_value)
          

    return render(request, 'nfile.html',locals())


def predict_sentiment(text):
  print("working")
  # Preprocess the text
  text = text.lower()
  text = re.sub(r'http\S+', '', text)
  text = re.sub(r'\W', ' ', text)
  tokens = word_tokenize(text)
  tokens = [word for word in tokens if word not in stopwords.words('english')]
  stems = [ps.stem(word) for word in tokens]
  preprocessed_text = ' '.join(stems)

  # Convert the preprocessed text to a sequence
  sequence = token.texts_to_sequences([preprocessed_text])
  sequence = pad_sequences(sequence, maxlen=max_sequence_length)

  # Predict the sentiment
  prediction = model.predict(sequence)[0]

  # Map the prediction to a sentiment label
  if np.argmax(prediction) == 0:
    sentiment = "Negative"
  elif np.argmax(prediction) == 1:
    sentiment = "Neutral"
  else:
    sentiment = "Positive"

  return sentiment


