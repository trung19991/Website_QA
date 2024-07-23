from flask import Flask, request, jsonify, send_from_directory
import torch
from transformers import XLMRobertaTokenizerFast, XLMRobertaForQuestionAnswering
from context_handler import ContextHandler  # Make sure this module is correctly implemented
import gdown
import os
import gc

app = Flask(__name__)

# Google Drive IDs for model and tokenizer
MODEL_FOLDER_ID = '1-6Dd1DNUAtHaLIoiEB5OT7eEoplhXF6Z'
TOKENIZER_FOLDER_ID = '1-G-YgQiAQ8hcKThj9Pi6FL6VDpjujE3W'

# Direct download URLs
MODEL_FOLDER_URL = f'https://drive.google.com/drive/folders/{MODEL_FOLDER_ID}'
TOKENIZER_FOLDER_URL = f'https://drive.google.com/drive/folders/{TOKENIZER_FOLDER_ID}'

# Paths to save model and tokenizer
model_path = './model'
tokenizer_path = './tokenizer'

# Create directories if they don't exist
os.makedirs(model_path, exist_ok=True)
os.makedirs(tokenizer_path, exist_ok=True)

# Download model and tokenizer if not present
if not os.path.exists(f'{model_path}/pytorch_model.bin'):
    gdown.download_folder(MODEL_FOLDER_URL, output=model_path, quiet=False, use_cookies=False)
    gc.collect()

if not os.path.exists(f'{tokenizer_path}/tokenizer.json'):
    gdown.download_folder(TOKENIZER_FOLDER_URL, output=tokenizer_path, quiet=False, use_cookies=False)
    gc.collect()

# Load tokenizer and model
tokenizer = XLMRobertaTokenizerFast.from_pretrained(tokenizer_path)
model = XLMRobertaForQuestionAnswering.from_pretrained(model_path)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Load context handler
squad_path = './squad.json'
context_handler = ContextHandler(squad_path)

def get_prediction(question):
    try:
        # Check if answer already exists
        answer = context_handler.get_answer_if_exists(question)
        if answer:
            return {'answer': answer}
        
        # Find best context
        context = context_handler.find_best_context(question)
        if context is None:
            return {'answer': "No valid context found."}
        
        # Tokenize and get model outputs
        inputs = tokenizer.encode_plus(question, context, return_tensors='pt', truncation=True, padding=True).to(device)
        outputs = model(**inputs)
        answer_start = torch.argmax(outputs.start_logits).item()
        answer_end = torch.argmax(outputs.end_logits).item() + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

        if answer.strip() == "":
            answer = "No answer found in the context."
        
        # Clean up memory
        del inputs
        del outputs
        gc.collect()
        
        return {'answer': answer}
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
