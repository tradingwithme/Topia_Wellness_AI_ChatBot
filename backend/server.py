import pandas as pd
import multiprocessing
from flask_cors import CORS
from flask import Flask, request, jsonify

from chatbot_entry import (
    get_mental_health_data,
    nltk_download,
    tokenize_dataset,
    load_checkpoint,
    generate_hybrid_response,
    save_approved_response,
    save_correction,
    fine_tune_model,
    fine_tune_with_approved
)

app = Flask(__name__)
CORS(app)  

print("Loading dataset and models...")
df = get_mental_health_data()
nltk_download()
tokenized_dataset, tfidf_vectorizer, tfidf_matrix = tokenize_dataset(df)
model, tokenizer = load_checkpoint()

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")

        if not user_message.strip():
            return jsonify({"response": "Please type a valid message."})

      
        response = generate_hybrid_response(
user_message,
df,
tfidf_vectorizer,
tfidf_matrix,
model,
tokenizer,
)

        return jsonify({"response": response})

    except Exception as e:
        print(f"Error in /api/chat: {e}")
        return jsonify({"response": "⚠️ Server error. Please try again."}), 500


if __name__ == "__main__": app.run(host="0.0.0.0", port=5000, debug=True)