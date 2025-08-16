import numpy as np
from transformers import pipeline
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

def summarize_responses(responses, summarizer, chunk_size=1024):
    """
    Summarize all valid retrieved responses using a HuggingFace model.
    Ensures all of combined_text is considered by chunking if needed.
    Args:
        responses (List[str]): List of response strings.
        summarizer (callable): HuggingFace pipeline for summarization.
        chunk_size (int): Max input size for the model (tokens/characters).
    Returns:
        str: Summarized response.
    """
    combined_text = " ".join(responses)
    # Split combined_text into chunks if it exceeds model input size
    chunks = [combined_text[i:i+chunk_size] for i in range(0, len(combined_text), chunk_size)]
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=128, min_length=30, do_sample=False)
        if isinstance(summary, list) and "summary_text" in summary[0]:
            summaries.append(summary[0]["summary_text"])
        else:
            summaries.append(str(summary))
    # If multiple summaries, summarize them again to condense
    if len(summaries) > 1:
        final_summary = summarizer(" ".join(summaries), max_length=128, min_length=30, do_sample=False)
        if isinstance(final_summary, list) and "summary_text" in final_summary[0]:
            return final_summary[0]["summary_text"]
        else:
            return str(final_summary)
    return summaries[0]

def generate_hybrid_response(
    user_input, df, tfidf_vectorizer, tfidf_matrix, model, tokenizer,
    response_encoder=None, question_encoder=None, sn_index=None, context_tuples=None,
    summarizer=None,
    retrieval_threshold=0.7, rule_threshold=0.5, top_k_neighbors=3
):
    """
    Hybrid response: retrieves responses using four techniques and summarizes them.
    Ensures all of combined_text is considered.
    """
    retrieved_responses = []

    # 1. Embedding-based retrieval using SimpleNeighbors
    if sn_index is not None and question_encoder is not None and context_tuples is not None:
        neighbors = retrieve_neighbors(user_input, sn_index, question_encoder, context_tuples, top_k=top_k_neighbors)
        retrieved_responses.extend([answer for _, answer in neighbors if answer])

    # 2. Retrieval-based response (TF-IDF)
    user_input_lemmatized = text_normalization(user_input)
    user_tfidf = tfidf_vectorizer.transform([user_input_lemmatized])
    cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix)
    most_similar_index = cosine_sim.argmax()
    retrieval_score = cosine_sim[0, most_similar_index]
    if retrieval_score > retrieval_threshold:
        retrieved_responses.append(df['Answers'].iloc[most_similar_index])

    # 3. Rule-based response
    rule_response_text = rule_based_response(user_input, df)
    rule_confidence = get_rule_based_confidence(user_input, df)
    if rule_confidence > rule_threshold and rule_response_text not in retrieved_responses:
        retrieved_responses.append(rule_response_text)

    # 4. Generative response (if retrieval and rule-based fail)
    #if not retrieved_responses:
    input_text = str(user_input)
    input_text = f"<HUMAN>: {input_text}\n<ASSISTANT>:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        input_ids,
        max_length=9999,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    generated_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_response = generated_response.replace(input_text, "").strip()
    retrieved_responses.append(generated_response)

    # Summarize all valid retrieved responses using the provided summarizer
    if summarizer is not None and len(retrieved_responses) > 1:
        return summarize_responses(retrieved_responses, summarizer)
    else:
        return retrieved_responses[0] if retrieved_responses else "I'm not sure how to respond to that. Can you please rephrase your question."
