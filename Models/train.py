import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

class EmotionalSupportDataset(torch.utils.data.Dataset):
    def __init__(self, dialog_pairs, tokenizer, max_length=512):
        self.dialog_pairs = dialog_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self):
        return len(self.dialog_pairs)

    def __getitem__(self, idx):
        dialog_text = self.dialog_pairs[idx][0]
        act = self.dialog_pairs[idx][1]
        emotion = self.dialog_pairs[idx][2]

        print(f"Original text: {dialog_text}")

        # Tokenize the dialogue text
        inputs = self.tokenizer(dialog_text, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = inputs.input_ids

        print(f"Tokenized input_ids: {input_ids}")

        # Create labels using act and emotion
        label_act = torch.tensor(act, dtype=torch.long)
        label_emotion = torch.tensor(emotion, dtype=torch.long)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': {
                'act': label_act,
                'emotion': label_emotion,
            },
        }


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        dialog_pairs = json.load(f)
    
    # Print the first few dialog pairs to inspect the data
    print("Loaded dialog pairs:")
    for dialog_pair in dialog_pairs[:5]:
        print(dialog_pair)
    
    # Ensure acts and emotions are integers
    for dialog_pair in dialog_pairs:
        dialog_pair[1] = int(dialog_pair[1])  # Convert act to integer
        dialog_pair[2] = int(dialog_pair[2])  # Convert emotion to integer

    return dialog_pairs

class CustomDataCollator:
    def __call__(self, features):
        input_ids = [feature['input_ids'] for feature in features]
        label_acts = [feature['labels']['act'] for feature in features]
        label_emotions = [feature['labels']['emotion'] for feature in features]

        return {
            'input_ids': torch.stack(input_ids, dim=0),
            'labels': {
                'act': torch.stack(label_acts, dim=0),
                'emotion': torch.stack(label_emotions, dim=0),
            },
        }

def main():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    train_data = load_data('data/train.json')
    val_data = load_data('data/validation.json')

    train_dataset = EmotionalSupportDataset(train_data, tokenizer)
    val_dataset = EmotionalSupportDataset(val_data, tokenizer)

    data_collator = CustomDataCollator()

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
        data_collator=data_collator  # Use the custom data collator here
    )

    trainer.train()
    model.save_pretrained('models/emotional_support_bot')
    tokenizer.save_pretrained('models/emotional_support_bot')

if __name__ == "__main__":
    main()
