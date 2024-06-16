# chatbot_interface.py
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved models and tokenizer
emotion_model = load_model('models/emotion_recognition_model.keras')

with open('data/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('data/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load preprocessed data to get the maxlen for padding
with open('data/preprocessed_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)
maxlen = X_train.shape[1]

# Function to predict emotion
def predict_emotion(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen)
    pred = emotion_model.predict(padded)
    emotion = label_encoder.inverse_transform([np.argmax(pred)])
    return emotion[0]

# Example chatbot response function
def chatbot_response(message):
    emotion = predict_emotion(message)
    if emotion == 'joy':
        response = "I'm glad to hear that! How can I assist you further?"
    elif emotion == 'anger':
        response = "I'm sorry you're feeling this way. Let's try to find a solution together."
    elif emotion == 'sadness':
        response = "I'm here for you. What's on your mind?"
    elif emotion == 'fear':
        response = "Take a deep breath. Let's work through this together."
    else:
        response = "Tell me more about what's going on."
    return response

