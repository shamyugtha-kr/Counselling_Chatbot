import torch

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
