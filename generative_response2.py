#generative_response.py
import numpy as np
from simpleneighbors import SimpleNeighbors
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

def compute_embeddings(texts, encoder):
    """
    Compute embeddings for a list of texts using the provided encoder.
    Args:
        texts (List[str]): List of input texts.
        encoder (callable): Function or model to compute embeddings.
    Returns:
        List: List of embedding vectors.
    """
    return encoder(texts)

def build_simpleneighbors_index(context_tuples, response_encoder):
    """
    Build a SimpleNeighbors index for all text, context tuples using response_encoder.
    Args:
        context_tuples (List[Tuple[str, str]]): List of (question, answer) pairs.
        response_encoder (callable): Encoder for responses.
    Returns:
        SimpleNeighbors: The built index.
    """
    responses = [resp for _, resp in context_tuples]
    embeddings = compute_embeddings(responses, response_encoder)
    dims = len(embeddings[0])
    sn = SimpleNeighbors(dims)
    for i, emb in enumerate(embeddings):
        sn.add_one(emb, i)
    sn.build()
    return sn

def retrieve_neighbors(user_input, sn_index, question_encoder, context_tuples, top_k=3):
    """
    Encode the question using question_encoder and retrieve nearest neighbors from the index.
    Args:
        user_input (str): The user's input question.
        sn_index (SimpleNeighbors): The built index.
        question_encoder (callable): Encoder for questions.
        context_tuples (List[Tuple[str, str]]): List of (question, answer) pairs.
        top_k (int): Number of neighbors to retrieve.
    Returns:
        List[Tuple[str, str]]: List of (question, answer) pairs.
    """
    user_emb = compute_embeddings([user_input], question_encoder)[0]
    nearest_indices = sn_index.nearest(user_emb, n=top_k)
    return [context_tuples[i] for i in nearest_indices]

def generate_hybrid_response(
    user_input, df, tfidf_vectorizer, tfidf_matrix, model, tokenizer,
    response_encoder=None, question_encoder=None, sn_index=None, context_tuples=None,
    retrieval_threshold=0.7, rule_threshold=0.5, top_k_neighbors=3
):
    """
    Enhanced hybrid response: includes embedding-based retrieval using SimpleNeighbors.
    """
    # 1. Embedding-based retrieval using SimpleNeighbors
    if sn_index is not None and question_encoder is not None and context_tuples is not None:
        neighbors = retrieve_neighbors(user_input, sn_index, question_encoder, context_tuples, top_k=top_k_neighbors)
        if neighbors:
            # Return the answer of the top neighbor
            return neighbors[0][1]
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
