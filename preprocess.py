import pandas as pd
import json

def preprocess_dataset(input_csv, output_json):
    df = pd.read_csv(input_csv)
    dialog_pairs = []
    
    for index, row in df.iterrows():
        dialog = row['dialog']
        acts_str = row['act'].strip()
        emotions_str = row['emotion'].strip()
        
        # Remove enclosing square brackets if they exist
        acts_str = acts_str.strip('[]')
        emotions_str = emotions_str.strip('[]')
        
        # Splitting space-separated values
        acts = acts_str.split()
        emotions = emotions_str.split()
        
        # Matching the lengths of acts and emotions lists
        min_len = min(len(acts), len(emotions))
        acts = acts[:min_len]
        emotions = emotions[:min_len]
        
        for act, emotion in zip(acts, emotions):
            dialog_pairs.append((dialog, int(act), int(emotion)))
    
    with open(output_json, 'w') as f:
        json.dump(dialog_pairs, f)

if __name__ == "__main__":
    preprocess_dataset('data/train.csv', 'data/train.json')
    preprocess_dataset('data/validation.csv', 'data/validation.json')
    preprocess_dataset('data/test.csv', 'data/test.json')
    
    print("Preprocessing complete.")
