from flask import request, jsonify
from .chatbot import Chatbot

chatbot = Chatbot()

def init_app(app):
    @app.route('/chat', methods=['POST'])
    def chat():
        user_input = request.json.get('message')
        response = chatbot.generate_response(user_input)
        return jsonify({'response': response})
