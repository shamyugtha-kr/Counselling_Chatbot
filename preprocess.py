import pandas as pd
import json
from ast import literal_eval

def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Convert string representation of lists to actual lists
    df['act'] = df['act'].apply(lambda x: literal_eval(x.replace(' ', ',')))
    df['emotion'] = df['emotion'].apply(lambda x: literal_eval(x.replace(' ', ',')))
    
    dialogs = df['dialog'].apply(eval).tolist()
    acts = df['act'].tolist()
    emotions = df['emotion'].tolist()
    
    return dialogs, acts, emotions

def preprocess_data():
    train_dialogs, train_acts, train_emotions = load_data('data/train.csv')
    test_dialogs, test_acts, test_emotions = load_data('data/test.csv')
    val_dialogs, val_acts, val_emotions = load_data('data/validation.csv')
    
    train_data = [{'dialog': dialog, 'act': act, 'emotion': emotion} for dialog, act, emotion in zip(train_dialogs, train_acts, train_emotions)]
    test_data = [{'dialog': dialog, 'act': act, 'emotion': emotion} for dialog, act, emotion in zip(test_dialogs, test_acts, test_emotions)]
    val_data = [{'dialog': dialog, 'act': act, 'emotion': emotion} for dialog, act, emotion in zip(val_dialogs, val_acts, val_emotions)]
    
    with open('data/preprocessed_train.json', 'w') as f:
        json.dump(train_data, f)
        
    with open('data/preprocessed_test.json', 'w') as f:
        json.dump(test_data, f)
        
    with open('data/preprocessed_val.json', 'w') as f:
        json.dump(val_data, f)
        
    print("Data preprocessing complete. Saved to JSON files.")

if __name__ == "__main__":
    preprocess_data()
