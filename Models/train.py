import json
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
from collator import CustomDataCollator

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    # Load preprocessed data
    train_data = load_json('data/preprocessed_train.json')
    val_data = load_json('data/preprocessed_val.json')

    # Convert data to Hugging Face datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # Load model
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_dir='./logs',
        evaluation_strategy="epoch"
    )

    # Instantiate custom data collator
    data_collator = CustomDataCollator()

    # Instantiate trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
