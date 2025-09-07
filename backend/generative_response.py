import numpy as np
from transformers import pipeline
from simpleneighbors import SimpleNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from model_handling import text_normalization
import boto3
import json

def get_retrieval_confidence(user_input, tfidf_vectorizer, tfidf_matrix):
    try:
        user_input_norm = text_normalization(user_input)
        user_tfidf = tfidf_vectorizer.transform([user_input_norm])
        cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix)
        return np.max(cosine_sim)
    except Exception as e:
        print(f"Error calculating retrieval confidence: {e}")
        return 0.0

def rule_based_response(user_input, df, combined_df_cv=None, stop_words=None):
    user_input_norm = text_normalization(user_input)
    keywords = user_input_norm.split()

    if combined_df_cv is None or stop_words is None:
        return "I'm not sure how to respond to that. Can you please rephrase your question?"

    for keyword in keywords:
        if keyword in combined_df_cv.columns and keyword not in stop_words:
            relevant_rows = df[combined_df_cv[keyword] > 0]
            if not relevant_rows.empty:
                return relevant_rows['Answers'].sample(1).iloc[0]
    return "I'm not sure how to respond to that. Can you please rephrase your question?"

def get_rule_based_confidence(user_input, df, combined_df_cv=None, stop_words=None):
    response = rule_based_response(user_input, df, combined_df_cv, stop_words)
    return 1.0 if response != "I'm not sure how to respond to that. Can you please rephrase your question?" else 0.0

def compute_embeddings(texts, encoder):
    return encoder(texts)

def build_simpleneighbors_index(context_tuples, response_encoder):
    responses = [resp for _, resp in context_tuples]
    embeddings = compute_embeddings(responses, response_encoder)
    dims = len(embeddings[0])
    sn = SimpleNeighbors(dims)
    for i, emb in enumerate(embeddings):
        sn.add_one(emb, i)
    sn.build()
    return sn

def retrieve_neighbors(user_input, sn_index, question_encoder, context_tuples, top_k=3):
    user_emb = compute_embeddings([user_input], question_encoder)[0]
    nearest_indices = sn_index.nearest(user_emb, n=top_k)
    return [context_tuples[i] for i in nearest_indices]

def sagemaker_summarize(text, endpoint_name):
    runtime = boto3.client('sagemaker-runtime')
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps({"inputs": text})
    )
    result = response['Body'].read().decode()
    summary = json.loads(result)
    if isinstance(summary, list) and "summary_text" in summary[0]:
        return summary[0]["summary_text"]
    elif isinstance(summary, dict) and "summary_text" in summary:
        return summary["summary_text"]
    return str(summary)

def summarize_responses(responses, summarizer=None, sagemaker_endpoint=None, chunk_size=1024):
    combined_text = " ".join(responses)
    chunks = [combined_text[i:i+chunk_size] for i in range(0, len(combined_text), chunk_size)]
    summaries = []

    for chunk in chunks:
        if sagemaker_endpoint:
            summary = sagemaker_summarize(chunk, sagemaker_endpoint)
        elif summarizer:
            result = summarizer(chunk, max_length=512, min_length=30, do_sample=False)
            summary = result[0]["summary_text"] if isinstance(result, list) else str(result)
        else:
            summary = chunk
        summaries.append(summary)

    if len(summaries) > 1:
        final_text = " ".join(summaries)
        if sagemaker_endpoint:
            return sagemaker_summarize(final_text, sagemaker_endpoint)
        elif summarizer:
            result = summarizer(final_text, max_length=512, min_length=30, do_sample=False)
            return result[0]["summary_text"] if isinstance(result, list) else str(result)
        else:
            return final_text
    return summaries[0]

def generate_hybrid_response(
    user_input, df, tfidf_vectorizer, tfidf_matrix, model, tokenizer,
    response_encoder=None, question_encoder=None, sn_index=None, context_tuples=None,
    summarizer=None, sagemaker_endpoint=None,
    retrieval_threshold=0.7, rule_threshold=0.5, top_k_neighbors=3
):
    retrieved_responses = []

    if sn_index and question_encoder and context_tuples:
        neighbors = retrieve_neighbors(user_input, sn_index, question_encoder, context_tuples, top_k=top_k_neighbors)
        retrieved_responses.extend([ans for _, ans in neighbors if ans])

    user_input_norm = text_normalization(user_input)
    user_tfidf = tfidf_vectorizer.transform([user_input_norm])
    cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix)
    most_similar_index = cosine_sim.argmax()
    if cosine_sim[0, most_similar_index] > retrieval_threshold:
        retrieved_responses.append(df['Answers'].iloc[most_similar_index])

    rule_response_text = rule_based_response(user_input, df)
    if get_rule_based_confidence(user_input, df) > rule_threshold and rule_response_text not in retrieved_responses:
        retrieved_responses.append(rule_response_text)

    input_text = f"<HUMAN>: {user_input}\n<ASSISTANT>:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        input_ids,
        max_length=9999,  # no truncation
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    gen_response = tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(input_text, "").strip()
    retrieved_responses.append(gen_response)

    if (summarizer or sagemaker_endpoint) and len(retrieved_responses) > 1:
        return summarize_responses(retrieved_responses, summarizer, sagemaker_endpoint)
    else:
        return retrieved_responses[0] if retrieved_responses else "I'm not sure how to respond to that. Can you please rephrase your question?"