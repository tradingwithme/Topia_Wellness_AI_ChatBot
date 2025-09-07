# Topia AI Wellness Chatbot

Thank you for granting me access to my project! 

This is my UX/UI/Fullstack project I worked on between February and April 2025. This server is intended to connect to the main website. Some tweaks have been made since then. A **full-stack AI chatbot** designed for mental health and wellness support. This project integrates **React.js frontend** with a **Python backend** leveraging **Hugging Face Transformers**, **PEFT adapters**, and optional **AWS SageMaker endpoints** for summarization and acceleration. The bot supports **retrieval, rule-based, and generative hybrid responses**, and it can **fine-tune iteratively using user feedback and corrections**.

---

## Table of Contents

* [Features](#features)
* [Architecture](#architecture)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Running the Application](#running-the-application)
* [Frontend](#frontend)
* [Backend](#backend)
* [Usage](#usage)
* [Feedback & Fine-Tuning Workflow](#feedback--fine-tuning-workflow)
* [SageMaker Integration](#sagemaker-integration)
* [Troubleshooting](#troubleshooting)
* [Contributing](#contributing)
* [License](#license)

---

## Features

* **Frontend** with animated backdrop.
* **Hybrid chatbot** combining:

  * Embedding-based retrieval (via SimpleNeighbors)
  * TF-IDF retrieval
  * Rule-based responses
  * Generative Transformer responses (DialoGPT-medium)
* **Iterative fine-tuning** using approved responses and corrections.
* **Multiprocessing** for background fine-tuning.
* **Dynamic summarization** using Hugging Face or optional SageMaker endpoint.
* **Adaptive model updates** without truncation, respecting full-length inputs.

---

## Architecture

```
┌───────────────┐           ┌───────────────────┐
│  React Frontend│ <-------> │  FastAPI / Flask  │
│ Chatbot UI    │           │ Backend API       │
└───────────────┘           └───────────────────┘
                                 │
                                 │
                     ┌───────────┴───────────┐
                     │  Response Generator    │
                     │ - TF-IDF Retrieval     │
                     │ - SimpleNeighbors      │
                     │ - Rule-based           │
                     │ - Generative (DialoGPT)│
                     └───────────┬───────────┘
                                 │
                     ┌───────────┴───────────┐
                     │  Feedback & Fine-Tune │
                     │ - Save approved CSV    │
                     │ - Save corrections CSV │
                     │ - Multiprocessing      │
                     └───────────────────────┘
```

---

## Prerequisites

* **Python >= 3.10**
* **Node.js >= 20**
* **npm or yarn**
* GPU is optional but recommended for faster fine-tuning.
* AWS account if using SageMaker endpoint (optional).

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/topia-ai-chatbot.git
cd topia-ai-chatbot
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd ../frontend
npm install
# or
yarn install
```

---

## Running the Application

### 1. Start Backend

```bash
cd backend
uvicorn chatbot_api:app --reload  # if using FastAPI
# OR
python chatbot_entry.py            # For CLI testing
```

### 2. Start Frontend

```bash
cd frontend
npm start
# Opens React app at http://localhost:3000
```

---

## Frontend

* **Framework:** React.js
* **Chat Features:**

  * Animated, majestic UI
  * Typing indicators
  * User-friendly input
  * Conditional buttons for approve/correction (excluded for welcome message)

**File:** `Chatbot.js`

* `messages` state tracks conversation.
* `handleSend` sends user input to backend API or generates default response.

---

## Backend

* **Framework:** FastAPI / Flask

* **Core Modules:**

  * `chatbot_entry.py` – main script for CLI and backend responses
  * `generative_response.py` – hybrid response generator
  * `csv_handling.py` – load and append approved responses
  * `model_handling.py` – tokenization, dataset preparation, PEFT model loading, fine-tuning

* **Key Concepts:**

  * Hybrid response combines TF-IDF retrieval, SimpleNeighbors embeddings, rule-based logic, and generative Transformer.
  * Multiprocessing allows background fine-tuning with minimal UI blocking.
  * Responses saved to `approved_responses.csv` for iterative improvements.

---

## Usage

* On first launch, the chatbot responds with a **welcome message**:
  `"Hello! I am your sophisticated chatbot. How can I assist you today?"`
* After user inputs:

  * Backend generates response
  * Approve/Correct buttons appear for user feedback (except for welcome message)
* Approved responses trigger periodic background fine-tuning.

---

## Feedback & Fine-Tuning Workflow

1. User approves response → saved in `approved_responses.csv`.
2. Every 5 approvals → background fine-tuning starts.
3. User corrects response → saved in `corrections.csv` → immediate fine-tuning.
4. Multiprocessing ensures **UI remains responsive** during fine-tuning.
5. No truncation: inputs processed fully using tokenizer (`max_length` set safely beyond model default).

---

## SageMaker Integration

* Optional: Use SageMaker endpoint for **summarization** or **accelerated inference**.
* Configure in `generative_response.py`:

```python
summarizer=None
sagemaker_endpoint="[endpoint goes here]"
bot_reply = generate_hybrid_response(
    user_input,
    df,
    tfidf_vectorizer,
    tfidf_matrix,
    model,
    tokenizer,
    summarizer=summarizer,
    sagemaker_endpoint=sagemaker_endpoint
)
```

* If no endpoint is provided, the Hugging Face pipeline is used.

---

## Troubleshooting

* **Invalid Date** on chat bubbles → Ensure `timestamp` in frontend is optional and handled correctly.
* **Module not found** errors → Run `pip install -r requirements.txt`.
* **WebSocket 404** → Ensure `npm start` is running on correct port.
* **Fine-tuning hangs** → Check multiprocessing permissions and model checkpoint availability.

---

## Contributing

* Fork the repository.
* Create a new branch for feature/bugfix.
* Submit a pull request with detailed description.

---

## License

MIT License © 2025