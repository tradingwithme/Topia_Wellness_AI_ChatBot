import os
import torch
from nltk import pos_tag
from peft import PeftModel
from datasets import Dataset
from nltk.stem import wordnet
from nltk import word_tokenize
from nltk import data, download
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer # Import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer # Import necessary transformers classes

def text_normalization(text):
  text = str(text).lower()
  spl_char_text = re.sub(r'[^ a-z]','',text)
  tokens = nltk.word_tokenize(spl_char_text)
  lema = wordnet.WordNetLemmatizer()
  tags_list = pos_tag(tokens, tagset=None)
  lema_words = []
  for token,pos_token in tags_list:
    if token not in stop:
      if pos_token.startswith('V'): pos_val='v'
      elif pos_token.startswith('J'): pos_val='a'
      elif pos_token.startswith('R'): pos_val='r'
      else: pos_val='n'
      lema_token = lema.lemmatize(token,pos_val)
      if lema_token not in stop: lema_words.append(lema_token)
  return " ".join(lema_words)

def format_data(row): return f"<HUMAN>: {row['Questions']}\n<ASSISTANT>: {row['Answers']}" # Format the question and answer as a conversation turn

def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=False, padding="max_length", max_length=9999)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

def nltk_download():
  try: data.find('tokenizers/punkt')
  except: download('punkt')
  try: data.find('corpora/stopwords')
  except: download('stopwords')
  try: data.find('taggers/averaged_perceptron_tagger')
  except: download('averaged_perceptron_tagger')
  try: data.find('tokenizers/punkt_tab')
  except: download('punkt_tab')
  try: data.find('taggers/averaged_perceptron_tagger_eng')
  except: download('averaged_perceptron_tagger_eng')
  try: data.find('corpora/wordnet')
  except: download('wordnet')

def tokenize_dataset():
  stop = stopwords.words('english')
  combined_df = get_mental_health_data()
  combined_df = combined_df[[combined_df.columns[:2]]]
  combined_df['lemmatized_text'] = combined_df['Questions'].apply(lambda x: text_normalization(x)) # Apply text normalization
  cv = CountVectorizer() # Create combined_df_cv using CountVectorizer (needed for rule_based_response)
  X = cv.fit_transform(combined_df['lemmatized_text'])
  combined_df_cv = pd.DataFrame(X.toarray(),columns=cv.get_feature_names_out())
  tfidf_vectorizer = TfidfVectorizer() # Initialize and fit the TF-IDF vectorizer (needed for retrieval_based_response)
  tfidf_matrix = tfidf_vectorizer.fit_transform(combined_df['lemmatized_text'])
  tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium") # Load the tokenizer
  if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token # Add a pad token if it's missing (necessary for batching)
  combined_df['text'] = combined_df.apply(format_data, axis=1)
  dataset = Dataset.from_pandas(combined_df[['text']]) # Convert the pandas DataFrame to a Hugging Face Dataset
  tokenized_dataset = dataset.map(tokenize_function, batched=True)
  print("Dataset successfully prepared and tokenized.")
  return tokenized_dataset

def load_checkpoint():
  model_dir_drive = "./mental_health_chatbot_model_iterative"
  print(f"Contents of {model_dir_drive}:")  # Verify the contents of the model directory
  try: print(os.listdir(model_dir_drive))
  except FileNotFoundError: print(f"Error: Directory not found at {model_dir_drive}. Please ensure the path is correct and Google Drive is mounted.")
  checkpoints = [d for d in os.listdir(model_dir_drive) if d.startswith("checkpoint-")]  # Find the latest checkpoint directory
  if checkpoints:
      latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1] # Sort checkpoints by step number to get the latest one
      checkpoint_path = os.path.join(model_dir_drive, latest_checkpoint)
      print(f"Latest checkpoint found at: {checkpoint_path}")
      print(f"Contents of {checkpoint_path}:") # Verify contents of the checkpoint directory
      try: print(os.listdir(checkpoint_path))
      except FileNotFoundError: print(f"Error: Checkpoint directory not found at {checkpoint_path}. Please verify the path.")
      try:
          base_model_name = "microsoft/DialoGPT-medium" # Base model name
          tokenizer = AutoTokenizer.from_pretrained(base_model_name)
          base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
          model = PeftModel.from_pretrained(base_model, checkpoint_path, device_map="cpu") # Load adapter onto CPU
          print(f"Successfully loaded PEFT model from checkpoint: {checkpoint_path} onto CPU.")
          return model, tokenizer
      except FileNotFoundError: raise Exception(f"Error: PEFT model files not found in {checkpoint_path}. Please verify the files exist.")
      except Exception as e: raise Exception(f"Failed to load PEFT model: {e}")
  else: raise Exception(f"No checkpoint directories found in {model_dir_drive}. Please ensure the training saved checkpoints correctly.")

def seetup_trainer(model, tokenizer, tokenized_dataset):
  training_args = TrainingArguments(
output_dir="./mental_health_chatbot_model_iterative",
overwrite_output_dir=True,
num_train_epochs=2, # Train for a small number of epochs in this iteration
per_device_train_batch_size=1, # Keep batch size small
gradient_accumulation_steps=4, # Accumulate gradients over steps
save_steps=50, # Save checkpoint more frequently for iterative training
save_total_limit=3, # Limit the number of saved checkpoints
logging_dir="./logs_iterative", # Directory for logs
logging_steps=10, # Log more frequently
learning_rate=5e-5, # Learning rate
fp16=False, # Disable mixed precision training for CPU/TPU
report_to="none" # Disable reporting to services like W&B for simplicity in iterative loops
)
  trainer = Trainer(
model=model,
args = training_args,
train_dataset=tokenized_dataset, # tokenized_dataset is assumed to be defined in a previous cell
)
  print("Trainer initialized successfully using standard Trainer.")
  return trainer, training_args

def train_dataset(trainer,training_args,trained_epochs,epochs_per_iteration):
  print(f"Starting training iteration. Trained epochs so far: {trained_epochs}/{total_epochs}")
  trainer.train()
  trained_epochs += epochs_per_iteration
  print(f"Finished training iteration. Total trained epochs: {trained_epochs}/{total_epochs}")
  return trainer, trained_epochs, epochs_per_iteration

def train_dataset_iterative(trainer,training_args,total_epochs=1):
  epochs_per_iteration = training_args.num_train_epochs
  if total_epochs < epochs_per_iteration:
    print("Total epochs are less then iterated epochs")
    total_epochs = epochs_per_iteration
  trained_epochs = 0
  trainer, trained_epochs, epochs_per_iteration = train_dataset(trainer,training_args,trained_epochs,epochs_per_iteration)
  while trained_epochs < total_epochs: trainer, trained_epochs, epochs_per_iteration = train_dataset(trainer,training_args,trained_epochs,epochs_per_iteration)
