import sys
import time
import pandas as pd
import multiprocessing
from csv_handling import get_mental_health_data
from generative_response import generate_hybrid_response
from model_handling import tokenize_dataset, load_checkpoint, setup_trainer, train_dataset_iterative

def fine_tune_model(tokenized_dataset, model, tokenizer):
    # Setup trainer and fine-tune for 1 epoch (or desired value)
    trainer = setup_trainer(model, tokenizer, tokenized_dataset)
    train_dataset_iterative(trainer, trainer.args, total_epochs=1)

def save_approved_response(user_input, response, filepath="mental_health_results.csv"):
    # Save approved Q&A to a CSV file
    approved_df = pd.DataFrame([{"Questions": user_input, "Answers": response}])
    try: approved_df.to_csv(filepath, 
mode='a', 
header=not pd.io.common.file_exists(filepath), 
index=False)
    except Exception as e: print(f"Error saving approved response: {e}")

def save_correction(user_input, corrected_response, filepath="corrections.csv"):
    correction_df = pd.DataFrame([{"Questions": user_input, "Answers": corrected_response}])
    try: correction_df.to_csv(filepath, mode='a', header=not pd.io.common.file_exists(filepath), index=False)
    except Exception as e: print(f"Error saving correction: {e}")

def fine_tune_with_approved():
    # Load approved responses dataset
    try:
        approved_df = pd.read_csv("approved_responses.csv")
        # You may need to tokenize and preprocess this dataset as required by your model
        # For example:
        tokenized_approved, _, _ = tokenize_dataset(approved_df)
        model, tokenizer = load_checkpoint()
        trainer = setup_trainer(model, tokenizer, tokenized_approved)
        train_dataset_iterative(trainer, trainer.args, total_epochs=1)
        # For demonstration, just print a message:
        print("Fine-tuning with approved responses...")
    except Exception as e: print(f"Error loading approved responses for fine-tuning: {e}")

def main():
    user_input = sys.argv[1] if len(sys.argv) > 1 else input("Enter your question: ")

    # Load data and models
    df = get_mental_health_data()
    tokenized_dataset, tfidf_vectorizer, tfidf_matrix = tokenize_dataset()
    model, tokenizer = load_checkpoint()

    # Generate response
    response = generate_hybrid_response(
        user_input,
        df,
        tfidf_vectorizer,
        tfidf_matrix,
        model,
        tokenizer
    )
    print(response)
    feedback = input("Do you approve this response? (y/n, or press Enter to skip): ").strip().lower()
    if feedback == "y":
        save_approved_response(user_input, response, filepath="approved_responses.csv")
        print("Approved response saved.")
        # Periodically fine-tune with approved responses (example: every 5 approvals)
        try:
            approved_df = pd.read_csv("approved_responses.csv")
            if len(approved_df) % 5 == 0:  # Fine-tune every 5 approved responses
                p = multiprocessing.Process(target=fine_tune_with_approved)
                p.start()
                print("Periodic fine-tuning with approved responses started in the background.")
        except Exception as e:
            print(f"Error checking approved responses for periodic fine-tuning: {e}")
    elif feedback == "n":
        corrected = input("Please provide a better answer: ").strip()
        if corrected:
            save_correction(user_input, corrected)
            print("Correction saved.")
            # Fine-tune immediately with corrections
            p = multiprocessing.Process(target=fine_tune_model, args=(tokenized_dataset, model, tokenizer))
            p.start()
            print("Model fine-tuning started in the background.")

if __name__ == "__main__": main()
