import json
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load the tokenizer globally
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    # Flatten the list of dialog sentences
    dialogs = [" ".join(dialog) for dialog in examples['dialog']]
    # Tokenize the dialogs
    tokenized_inputs = tokenizer(dialogs, padding="max_length", truncation=True, max_length=512)

    # Ensure labels are converted to tensors
    tokenized_inputs["labels"] = [emotion[0] if isinstance(emotion, list) else emotion for emotion in examples["emotion"]]
    return tokenized_inputs

def main():
    # Load preprocessed data
    train_data = load_json('data/preprocessed_train.json')
    test_data = load_json('data/preprocessed_test.json')
    val_data = load_json('data/preprocessed_val.json')
    
    # Convert the data to Dataset format
    train_dataset = Dataset.from_dict({"dialog": [example['dialog'] for example in train_data],
                                       "emotion": [example['emotion'] for example in train_data]})
    test_dataset = Dataset.from_dict({"dialog": [example['dialog'] for example in test_data],
                                      "emotion": [example['emotion'] for example in test_data]})
    val_dataset = Dataset.from_dict({"dialog": [example['dialog'] for example in val_data],
                                     "emotion": [example['emotion'] for example in val_data]})
    
    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Tokenize the datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",  # Use eval_strategy instead of evaluation_strategy
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    
    # Define data collator
    data_collator = DataCollatorWithPadding(tokenizer)

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()
