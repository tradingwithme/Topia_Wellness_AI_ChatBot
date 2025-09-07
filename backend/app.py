from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import multiprocessing

from csv_handling import get_mental_health_data
from generative_response import generate_hybrid_response
from model_handling import (
    nltk_download,
    tokenize_dataset,
    load_checkpoint,
    setup_trainer,
    train_dataset_iterative,
)

def fine_tune_model(tokenized_dataset, model, tokenizer):
    """Fine-tune the model on a tokenized dataset."""
    trainer = setup_trainer(model, tokenizer, tokenized_dataset)
    train_dataset_iterative(trainer, trainer.args, total_epochs=1)

def save_approved_response(user_input, response, filepath="approved_responses.csv"):
    """Save approved Q&A to CSV for reinforcement training."""
    approved_df = pd.DataFrame([{"Questions": user_input, "Answers": response}])
    try:
        approved_df.to_csv(
            filepath,
            mode='a',
            header=not pd.io.common.file_exists(filepath),
            index=False
        )
    except Exception as e:
        print(f"Error saving approved response: {e}")

def save_correction(user_input, corrected_response, filepath="corrections.csv"):
    """Save user-corrected answers to CSV."""
    correction_df = pd.DataFrame([{"Questions": user_input, "Answers": corrected_response}])
    try:
        correction_df.to_csv(
            filepath,
            mode='a',
            header=not pd.io.common.file_exists(filepath),
            index=False
        )
    except Exception as e:
        print(f"Error saving correction: {e}")

def fine_tune_with_approved():
    """Fine-tune on all approved responses periodically."""
    try:
        approved_df = pd.read_csv("approved_responses.csv")
        tokenized_approved, _, _ = tokenize_dataset(approved_df)
        model, tokenizer = load_checkpoint()
        trainer = setup_trainer(model, tokenizer, tokenized_approved)
        train_dataset_iterative(trainer, trainer.args, total_epochs=1)
        print("Fine-tuned with approved responses")
    except Exception as e:
        print(f"Error loading approved responses for fine-tuning: {e}")

app = Flask(__name__)
CORS(app)

print("Loading data & models...")
df = get_mental_health_data()
nltk_download()
tokenized_dataset, tfidf_vectorizer, tfidf_matrix = tokenize_dataset(df)
model, tokenizer = load_checkpoint()
print("Backend ready ✅")

@app.route("/chat", methods=["POST"])
def chat():
    """Generate AI response for user input."""
    data = request.get_json()
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    response = generate_hybrid_response(
user_input,
df,
tfidf_vectorizer,
tfidf_matrix,
model,
tokenizer
)

    return jsonify({"response": response})

@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Handle user feedback (approve or correct).
    Expected payload:
      {
        "user_input": "...",
        "response": "...",
        "feedback": "y" | "n",
        "correction": "..." (if applicable)
      }
    """
    data = request.get_json()
    user_input = data.get("user_input", "")
    response = data.get("response", "")
    feedback_val = data.get("feedback", "").lower()
    correction = data.get("correction", "")

    if feedback_val == "y":
        save_approved_response(user_input, response)
        try:
            approved_df = pd.read_csv("approved_responses.csv")
            if len(approved_df) % 5 == 0:
                p = multiprocessing.Process(target=fine_tune_with_approved)
                p.start()
                print("Started background fine-tuning with approved responses.")
        except Exception as e:
            print(f"Error checking approvals: {e}")
        return jsonify({"status": "approved", "message": "Response approved."})

    elif feedback_val == "n":
        if correction:
            save_correction(user_input, correction)
            p = multiprocessing.Process(
                target=fine_tune_model,
                args=(tokenized_dataset, model, tokenizer)
            )
            p.start()
            print("Background fine-tuning started with corrections.")
            return jsonify({"status": "corrected", "message": "Correction saved & fine-tuning started."})
        else:
            return jsonify({"status": "rejected", "message": "Correction missing."}), 400

    return jsonify({"status": "skipped", "message": "Feedback skipped."})

if __name__ == "__main__": app.run(port=5000, debug=True)