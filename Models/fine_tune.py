# fine_tune_model.py
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Load the dataset
dataset = load_dataset('empathetic_dialogues', split='train')

# Initialize the model and tokenizer
model_name = 'gpt2-medium'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples['utterance'], padding='max_length', truncation=True, max_length=128)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Fine-tune the model
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained('models/counseling_gpt2')
tokenizer.save_pretrained('models/counseling_gpt2')