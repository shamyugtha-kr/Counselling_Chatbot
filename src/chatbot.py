# src/chatbot.py
import joblib

class EmotionalCounselorBot:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict_emotion(self, text):
        return self.model.predict([text])[0]

    def generate_response(self, emotion):
        responses = {
            'happy': "I'm glad to hear that! What made you happy today?",
            'sad': "I'm sorry you're feeling this way. Do you want to talk about it?",
            'angry': "It sounds like something is bothering you. Can you tell me more?",
            'neutral': "I'm here to listen. Tell me more about what's on your mind."
        }
        return responses.get(emotion, "I'm here to listen.")

    def respond(self, text):
        emotion = self.predict_emotion(text)
        return self.generate_response(emotion)

if __name__ == "__main__":
    bot = EmotionalCounselorBot('../models/emotion_model.pkl')
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = bot.respond(user_input)
        print(f"Bot: {response}")
