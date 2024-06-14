import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

class EmotionalSupportDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dialog, acts, emotions = self.data[idx]
        input_text = ' '.join(dialog)
        target_text = ' '.join(dialog)
        input_ids = self.tokenizer.encode(input_text, truncation=True, max_length=self.max_length)
        target_ids = self.tokenizer.encode(target_text, truncation=True, max_length=self.max_length)
        return torch.tensor(input_ids), torch.tensor(target_ids)

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def main():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    train_data = load_data('data/preprocessed_train.json')
    val_data = load_data('data/preprocessed_val.json')
    
    train_dataset = EmotionalSupportDataset(train_data, tokenizer)
    val_dataset = EmotionalSupportDataset(val_data, tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    model.save_pretrained('models/emotional_support_bot')
    tokenizer.save_pretrained('models/emotional_support_bot')

if __name__ == "__main__":
    main()
