import pandas as pd
import json

def load_data(file_path):
    df = pd.read_csv(file_path)
    dialogs = df['dialog'].apply(eval).tolist()
    acts = df['act'].apply(eval).tolist()
    emotions = df['emotion'].apply(eval).tolist()
    return dialogs, acts, emotions

def preprocess_data():
    train_dialogs, train_acts, train_emotions = load_data('data/train.csv')
    test_dialogs, test_acts, test_emotions = load_data('data/test.csv')
    val_dialogs, val_acts, val_emotions = load_data('data/validation.csv')
    
    train_data = list(zip(train_dialogs, train_acts, train_emotions))
    test_data = list(zip(test_dialogs, test_acts, test_emotions))
    val_data = list(zip(val_dialogs, val_acts, val_emotions))
    
    with open('data/preprocessed_train.json', 'w') as f:
        json.dump(train_data, f)
        
    with open('data/preprocessed_test.json', 'w') as f:
        json.dump(test_data, f)
        
    with open('data/preprocessed_val.json', 'w') as f:
        json.dump(val_data, f)
        
    print("Data preprocessing complete. Saved to JSON files.")

if __name__ == "__main__":
    preprocess_data()
