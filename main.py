#Step 1: IMport Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index= {value: key for key, value in word_index.items()}

#Load the pre trained model with ReLU activation
from tensorflow.keras.models import load_model

model = load_model('simple_rnn_imdb.keras')

#Step 2:Helper Functions
#Function to decode reviews
def decode_review(encoded_review):
  return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])
  
#function to preprocess user input
import re

def preprocess_text(text):
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)

    words = text.split()

    encoded_review = [
        word_index.get(word, 2) 
        if word_index.get(word, 2) < 10000 
        else 2
        for word in words
    ]

    return sequence.pad_sequences([encoded_review], maxlen=200)

##Step 3: Prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)

    threshold = 0.69
    sentiment = 'Positive' if prediction[0][0] > threshold else 'Negative'

    return sentiment, prediction[0][0]



#streamlit app
import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

#User input
user_input= st.text_area('Movie Review')

if st.button('Classify'):
  if user_input.strip() == "":
    st.warning("Please enter a movie review.")
  else:
    sentiment, score = predict_sentiment(user_input)

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {score}')




