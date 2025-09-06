import argparse
import pandas as pd
import multiprocessing
from csv_handling import get_mental_health_data
from generative_response import generate_hybrid_response
from model_handling import nltk_download, tokenize_dataset, load_checkpoint, setup_trainer, train_dataset_iterative, fine_tune_model

# ---------------------------
# Save approved responses / corrections
# ---------------------------
def save_approved_response(user_input, response, filepath="approved_responses.csv"):
    approved_df = pd.DataFrame([{"Questions": user_input, "Answers": response}])
    try:
        approved_df.to_csv(filepath, mode='a', header=not pd.io.common.file_exists(filepath), index=False)
    except Exception as e:
        print(f"Error saving approved response: {e}")

def save_correction(user_input, corrected_response, filepath="corrections.csv"):
    correction_df = pd.DataFrame([{"Questions": user_input, "Answers": corrected_response}])
    try:
        correction_df.to_csv(filepath, mode='a', header=not pd.io.common.file_exists(filepath), index=False)
    except Exception as e:
        print(f"Error saving correction: {e}")

# ---------------------------
# Fine-tuning with approved responses
# ---------------------------
def fine_tune_with_approved():
    try:
        approved_df = pd.read_csv("approved_responses.csv")
        tokenized_approved, _, _ = tokenize_dataset()
        model, tokenizer = load_checkpoint()
        trainer = setup_trainer(model, tokenizer, tokenized_approved)
        train_dataset_iterative(trainer, total_epochs=1)
        print("Fine-tuning with approved responses complete.")
    except Exception as e:
        print(f"Error fine-tuning with approved responses: {e}")

# ---------------------------
# Main interactive loop
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feedback', type=str, default=None)
    parser.add_argument('--user_input', type=str, default=None)
    parser.add_argument('--response', type=str, default=None)
    args = parser.parse_args()

    # Load dataset and model
    nltk_download()
    df = get_mental_health_data()
    tokenized_dataset, tfidf_vectorizer, tfidf_matrix = tokenize_dataset()
    model, tokenizer = load_checkpoint()

    # Generate welcome message (no buttons)
    bot_reply = "Welcome, to Topia's AI Wellness Chatbot! How may I assist you today?"
    print(bot_reply)

    # Interactive input
    while True:
        user_input = args.user_input or input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot.")
            break

        # Generate hybrid response
        response = generate_hybrid_response(
            user_input,
            df,
            tfidf_vectorizer,
            tfidf_matrix,
            model,
            tokenizer
        )
        print(f"Bot: {response}")

        # Feedback handling
        feedback = input("Do you approve this response? (y/n, Enter to skip): ").strip().lower()
        if feedback == "y" or (args.feedback and args.feedback == 'y'):
            save_approved_response(user_input, response)
            print("Approved response saved.")

            # Fine-tune every 5 approved responses
            try:
                approved_df = pd.read_csv("approved_responses.csv")
                if len(approved_df) % 5 == 0:
                    p = multiprocessing.Process(target=fine_tune_with_approved)
                    p.start()
                    print("Background fine-tuning started.")
            except Exception as e:
                print(f"Error checking approved responses for fine-tuning: {e}")

        elif feedback == "n" or (args.feedback and args.feedback == 'n'):
            corrected = input("Please provide a better answer: ").strip()
            if corrected:
                save_correction(user_input, corrected)
                print("Correction saved.")

                # Fine-tune immediately in the background
                p = multiprocessing.Process(target=fine_tune_model, args=(tokenized_dataset, model, tokenizer))
                p.start()
                print("Background fine-tuning with correction started.")

if __name__ == "__main__": main()
