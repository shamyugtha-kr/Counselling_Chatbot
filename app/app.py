from flask import Flask, render_template, request, jsonify
from chatbot import chatbot_response

# Initialize Flask app
app = Flask(__name__, template_folder='../templates')  # Adjust the template_folder path

# Route to serve the index.html page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle POST requests from the chat form
@app.route('/get_response', methods=['POST'])
def get_response():
    # Ensure 'message' key is in the POST request
    if 'message' in request.form:
        message = request.form['message']
        response = chatbot_response(message)
        return jsonify({'response': response})
    else:
        return jsonify({'response': 'Error: No message provided.'})

if __name__ == "__main__":
    app.run(debug=True)
