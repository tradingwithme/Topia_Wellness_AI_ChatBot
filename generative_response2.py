import numpy as np
from model_handling import text_normalization
from sklearn.metrics.pairwise import cosine_similarity

def get_retrieval_confidence(user_input, tfidf_vectorizer, tfidf_matrix):
    """
    Calculates the confidence score for a retrieval-based response.

    Args:
        user_input (str): The user's input query.
        tfidf_vectorizer: The fitted TfidfVectorizer.
        tfidf_matrix: The TF-IDF matrix of the questions.

    Returns:
        float: The cosine similarity score (confidence) or 0 if an error occurs.
    """
    try:
        user_input_lemmatized = text_normalization(user_input)
        user_tfidf = tfidf_vectorizer.transform([user_input_lemmatized])
        cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix)
        return np.max(cosine_sim) # Return the highest cosine similarity score
    except Exception as e:
        print(f"Error calculating retrieval confidence: {e}")
        return 0.0

def rule_based_response(user_input, df):
    """
    Generates a response based on predefined rules and keywords from the dataframe.

    Args:
        user_input (str): The user's input query.
        df (pd.DataFrame): The dataframe containing questions and answers.

    Returns:
        str: A generated response or a default message.
    """
    user_input_lemmatized = text_normalization(user_input)
    keywords = user_input_lemmatized.split()

    for keyword in keywords:
        if keyword in combined_df_cv.columns and keyword not in stop:
            relevant_rows = df[combined_df_cv[keyword] > 0]
            if not relevant_rows.empty:
                return relevant_rows['Answers'].sample(1).iloc[0]

    return "I'm not sure how to respond to that. Can you please rephrase your question?"

def get_rule_based_confidence(user_input, df):
    """
    Determines the confidence score for a rule-based response.

    Args:
        user_input (str): The user's input query.
        df (pd.DataFrame): The dataframe containing questions and answers.

    Returns:
        float: 1.0 if a rule-based response is found, 0.0 otherwise.
    """
    rule_response = rule_based_response(user_input, df)  # You can refine this based on the complexity or certainty of your rules.
    if rule_response != "I'm not sure how to respond to that. Can you please rephrase your question?": return 1.0
    else: return 0.0

def generate_hybrid_response(user_input, df, tfidf_vectorizer, tfidf_matrix, model, tokenizer, retrieval_threshold=0.7, rule_threshold=0.5):
    """
    Generates a response using a hybrid approach (retrieval, rule-based, and generative).

    Args:
        user_input (str): The user's input query.
        df (pd.DataFrame): The dataframe containing questions and answers.
        tfidf_vectorizer: The fitted TfidfVectorizer.
        tfidf_matrix: The TF-IDF matrix of the questions.
        model: The fine-tuned language model.
        tokenizer: The tokenizer for the language model.
        retrieval_threshold (float): The cosine similarity threshold for retrieval-based response.
        rule_threshold (float): The confidence threshold for rule-based response.


    Returns:
        str: The generated response.
    """
    # 1. Retrieval-based response
    user_input_lemmatized = text_normalization(user_input)
    user_tfidf = tfidf_vectorizer.transform([user_input_lemmatized])
    cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix)
    most_similar_index = cosine_sim.argmax()
    retrieval_score = cosine_sim[0, most_similar_index]
    retrieval_response = df['Answers'].iloc[most_similar_index] if retrieval_score > retrieval_threshold else None

    # 2. Rule-based response
    rule_response_text = rule_based_response(user_input, df)
    rule_confidence = get_rule_based_confidence(user_input, df) # Use the confidence function
    rule_response = rule_response_text if rule_confidence > rule_threshold else None # Use the threshold

    # 3. Select the best response based on confidence/thresholds
    if retrieval_response is not None and (rule_response is None or retrieval_score >= rule_confidence): return retrieval_response
    elif rule_response is not None: return rule_response
    else:
        # 4. Generative response (if retrieval and rule-based fail)
        input_text = str(user_input) # Ensure the user input is a string before encoding
        input_text = f"<HUMAN>: {input_text}\n<ASSISTANT>:" # Format the input for the generative model as a conversation turn
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device) # Encode the input and ensure it's on the correct device
        output_ids = model.generate(  # Generate a response using the fine-tuned model
            input_ids,
            max_length=9999,  # Adjust max length as needed
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,  # Prevent repeating n-grams
            early_stopping=True  # Stop when the generation is complete
        )
        generated_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)  # Decode the generated response and remove the input text
        generated_response = generated_response.replace(input_text, "").strip()  # Remove the input prompt from the generated text
        return generated_response
