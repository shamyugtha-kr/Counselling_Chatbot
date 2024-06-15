import pandas as pd
import ast

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Convert string representations of lists into actual lists
    df['dialog'] = df['dialog'].apply(ast.literal_eval)
    df['act'] = df['act'].apply(lambda x: list(map(int, x.strip('[]').split())))
    df['emotion'] = df['emotion'].apply(lambda x: list(map(int, x.strip('[]').split())))

    # Flatten the dataframe
    dialogs, acts, emotions = [], [], []
    for i in range(len(df)):
        dialog = df.iloc[i]['dialog']
        act = df.iloc[i]['act']
        emotion = df.iloc[i]['emotion']
        
        if len(dialog) != len(act) or len(act) != len(emotion):
            print(f"Mismatch in lengths at index {i}:")
            print(f"Dialog length: {len(dialog)}, Act length: {len(act)}, Emotion length: {len(emotion)}")
            continue
        
        dialogs.extend(dialog)
        acts.extend(act)
        emotions.extend(emotion)

    # Check lengths before creating the DataFrame
    if len(dialogs) != len(acts) or len(dialogs) != len(emotions):
        raise ValueError("Flattened lists do not have the same length")

    # Ensure all dialog entries are strings
    dialogs = [str(dialog) for dialog in dialogs]
    
    flattened_df = pd.DataFrame({'text': dialogs, 'act': acts, 'emotion': emotions})
    
    # Example preprocessing steps (customize as needed)
    flattened_df = flattened_df.dropna()
    flattened_df['text'] = flattened_df['text'].str.lower()
    return flattened_df

if __name__ == "__main__":
    train_df = load_data('data/train.csv')
    validation_df = load_data('data/validation.csv')
    test_df = load_data('data/test.csv')

    train_df = preprocess_data(train_df)
    validation_df = preprocess_data(validation_df)
    test_df = preprocess_data(test_df)

    train_df.to_csv('data/train_preprocessed.csv', index=False)
    validation_df.to_csv('data/validation_preprocessed.csv', index=False)
    test_df.to_csv('data/test_preprocessed.csv', index=False)
