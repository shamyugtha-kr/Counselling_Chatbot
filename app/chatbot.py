from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class Chatbot:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained('models/emotional_support_bot')
        self.tokenizer = GPT2Tokenizer.from_pretrained('models/emotional_support_bot')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def generate_response(self, user_input):
        input_ids = self.tokenizer.encode(user_input, return_tensors='pt').to(self.device)
        chat_history_ids = self.model.generate(input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
        return response
