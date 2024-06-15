# src/train_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def load_data(file_path):
    return pd.read_csv(file_path)

if __name__ == "__main__":
    train_df = load_data('../data/train_preprocessed.csv')
    validation_df = load_data('../data/validation_preprocessed.csv')

    vectorizer = TfidfVectorizer()
    model = LogisticRegression()

    X_train = train_df['text']
    y_train = train_df['emotion']
    X_val = validation_df['text']
    y_val = validation_df['emotion']

    pipeline = make_pipeline(vectorizer, model)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f'Validation Accuracy: {accuracy}')

    # Save the model
    import joblib
    joblib.dump(pipeline, '../models/emotion_model.pkl')
