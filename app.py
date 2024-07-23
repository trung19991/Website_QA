from flask import Flask, request, jsonify, send_from_directory
from context_handler import ContextHandler
import os

app = Flask(__name__)

# Đường dẫn lưu mô hình và tokenizer
model_path = './model'
tokenizer_path = './tokenizer'

# Load ContextHandler
squad_path = './squad.json'
context_handler = ContextHandler(squad_path)

def get_prediction(question):
    try:
        # Check if question exists in squad.json
        answer = context_handler.get_answer_if_exists(question)
        if answer:
            return {'answer': answer}
        
        # Placeholder response for now
        return {'answer': "Model and tokenizer are temporarily disabled."}
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {'answer': "Error during prediction"}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        question = data['question']
        result = get_prediction(question)
        return jsonify(result)
    except Exception as e:
        print(f"Error in /predict route: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

@app.route('/contexts', methods=['GET'])
def contexts():
    try:
        contexts = context_handler.contexts
        return jsonify({'contexts': contexts})
    except Exception as e:
        print(f"Error in /contexts route: {e}")
        return jsonify({'error': 'An error occurred while fetching contexts.'}), 500

@app.route('/similar_questions', methods=['POST'])
def similar_questions():
    try:
        data = request.json
        input_question = data['question']
        similar_questions = context_handler.find_similar_questions(input_question)
        return jsonify({'similar_questions': similar_questions})
    except Exception as e:
        print(f"Error in /similar_questions route: {e}")
        return jsonify({'error': 'An error occurred while fetching similar questions.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
