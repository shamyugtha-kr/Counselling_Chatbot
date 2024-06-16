from flask import Flask, render_template, request, jsonify
from chatbot import chatbot_response

# Initialize Flask app
app = Flask(__name__, template_folder='../templates')  # Adjust the template_folder path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    message = request.form['message']
    response = chatbot_response(message)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)

