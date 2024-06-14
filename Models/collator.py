import torch

class CustomDataCollator:
    def __call__(self, features):
        dialogues = [feature['dialog'] for feature in features]
        acts = [feature['act'] for feature in features]
        emotions = [feature['emotion'] for feature in features]

        # Convert to tensors
        dialogues = torch.tensor(dialogues, dtype=torch.long)
        acts = torch.tensor(acts, dtype=torch.long)
        emotions = torch.tensor(emotions, dtype=torch.long)

        return {
            'input_ids': dialogues,
            'labels': acts,
            'emotions': emotions,
        }
