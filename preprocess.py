import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
data = pd.read_csv('dataset.csv')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = re.sub(r'\s+', ' ', text, flags=re.I)  # Remove all extra spaces
    text = text.lower()  # Convert to lower case
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords.words('english')]
    text = ' '.join(text)
    return text

# Apply preprocessing
data['content'] = data['content'].apply(preprocess_text)

# Encode the labels
label_encoder = LabelEncoder()
data['sentiment'] = label_encoder.fit_transform(data['sentiment'])

# Prepare the tokenizer
tokenizer = Tokenizer(num_words=5000, lower=True, split=' ')
tokenizer.fit_on_texts(data['content'].values)

# Convert text to sequences and pad them
X = tokenizer.texts_to_sequences(data['content'].values)
X = pad_sequences(X)

# Prepare labels
y = data['sentiment'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)