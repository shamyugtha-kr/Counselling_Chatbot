# fine_tune_model.py
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Load the dataset with trust_remote_code=True
dataset = load_dataset('empathetic_dialogues', split='train[:1%]', trust_remote_code=True)

# Initialize the model and tokenizer
model_name = 'gpt2'  # Switch to the smaller GPT-2 model
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add padding token to tokenizer if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    return tokenizer(examples['utterance'], padding='max_length', truncation=True, max_length=128)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Fine-tune the model
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,  # Reduce number of epochs for faster training
    per_device_train_batch_size=2,  # Reduce batch size
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    fp16=True,  # Enable mixed precision training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained('models/counseling_gpt2')
tokenizer.save_pretrained('models/counseling_gpt2')
