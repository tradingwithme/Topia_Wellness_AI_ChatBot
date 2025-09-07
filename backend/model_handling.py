import os
import re
from nltk import pos_tag, word_tokenize, data, download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from peft import PeftModel
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def text_normalization(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z ]', '', text)
    tokens = word_tokenize(text)
    normalized = []
    for token, pos_tag_ in pos_tag(tokens):
        if token in stop_words:
            continue
        if pos_tag_.startswith('V'):
            pos = 'v'
        elif pos_tag_.startswith('J'):
            pos = 'a'
        elif pos_tag_.startswith('R'):
            pos = 'r'
        else:
            pos = 'n'
        lemma = lemmatizer.lemmatize(token, pos)
        if lemma not in stop_words:
            normalized.append(lemma)
    return " ".join(normalized)

def format_data(row):
    return f"<HUMAN>: {row['Questions']}\n<ASSISTANT>: {row['Answers']}"

def tokenize_function(examples, tokenizer):
    tokens = tokenizer(examples["text"], truncation=False, padding="max_length", max_length=9999)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

def nltk_download():
    for resource in ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']:
        try:
            data.find(resource)
        except:
            download(resource)

def tokenize_dataset(df, tokenizer):
    df['lemmatized_text'] = df['Questions'].apply(text_normalization)
    df['text'] = df.apply(format_data, axis=1)
    dataset = Dataset.from_pandas(df[['text']])
    tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    print("Dataset successfully tokenized.")
    return tokenized_dataset

def load_checkpoint(model_dir="./mental_health_chatbot_model_iterative", base_model_name="microsoft/DialoGPT-medium"):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if os.path.exists(model_dir) and any(d.startswith("checkpoint-") for d in os.listdir(model_dir)):
        checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
        checkpoint_path = os.path.join(model_dir, latest_checkpoint)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, checkpoint_path, device_map="cpu")
        print(f"Loaded PEFT model from checkpoint: {checkpoint_path}")
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        print(f"No checkpoint found. Initialized new model from {base_model_name}.")
    return model, tokenizer

def setup_trainer(model, tokenizer, tokenized_dataset):
    training_args = TrainingArguments(
        output_dir="./mental_health_chatbot_model_iterative",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        save_steps=50,
        save_total_limit=3,
        logging_dir="./logs_iterative",
        logging_steps=10,
        learning_rate=5e-5,
        fp16=False,
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    print("Trainer initialized successfully.")
    return trainer, training_args

def train_dataset(trainer, training_args, trained_epochs, epochs_per_iteration, total_epochs):
    while trained_epochs < total_epochs:
        print(f"Training iteration: epochs {trained_epochs}/{total_epochs}")
        trainer.train()
        trained_epochs += epochs_per_iteration
    print(f"Training complete. Total epochs trained: {trained_epochs}")
    return trainer

def train_dataset_iterative(trainer, training_args, total_epochs=1):
    return train_dataset(trainer, training_args, trained_epochs=0, epochs_per_iteration=training_args.num_train_epochs, total_epochs=total_epochs)
