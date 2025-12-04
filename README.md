🎧 YouTube Chinese Sentiment Analyzer

AI-driven NLP pipeline for understanding polarized Chinese discussions

| Negative Cloud <img width="300" height="500" alt="image" src="https://github.com/user-attachments/assets/096c0f39-c771-4d54-a070-0afde21b9d22" />
| Positive Cloud <img width="300" height="500" alt="image" src="https://github.com/user-attachments/assets/2c103a63-10e7-4a1e-8b5f-e7df3d46f013" />
🚀 Overview

A full end-to-end Chinese NLP sentiment analysis system that:

collects YouTube comments (including replies) via API

extracts keywords and processes Chinese text

performs 6-level sentiment classification with a fine-tuned BERT model

identifies positive vs. negative narratives

visualizes each side as a Taiwan-shaped word cloud

This tool works especially well for highly polarized Chinese-language videos
(e.g., political debates, controversies, social issues, movie reviews).

✨ Features

Fetch top-level comments and replies automatically

Chinese preprocessing: character filtering + jieba tokenization

Fine-tuned BERT for nuanced sentiment scoring (1–5)

Automatic polarity grouping (positive vs. negative clusters)

Taiwan-shaped word clouds that highlight dominant themes

Summarize thousands of comments into an interpretable visual snapshot

🧠 Architecture
YouTube Video URL
        ↓
YouTube Data API
        ↓
Raw Comments + Replies
        ↓
Chinese Preprocessing
(jieba tokenization, keyword extraction)
        ↓
BERT Sentiment Model
(6-level classification)
        ↓
Polarity Separation
(positive / negative)
        ↓
Word Frequency Analysis
        ↓
Taiwan-shaped Word Clouds

🛠 Tech Stack

Python

PyTorch, HuggingFace Transformers

YouTube Data API v3

jieba, wordcloud, matplotlib, Pillow

tqdm, numpy, pandas

JSON training dataset (trainword.json, excluded from repo)

📁 Project Structure
Hiproject/
│── hiproject.py          # main script (end-to-end pipeline)
│── extract.py            # optional: preprocessing utilities
│── trained_model/        # fine-tuned BERT model (excluded)
│── taiwan.jpg            # shape mask for visualizations
│── requirements.txt      # package dependencies
│── README.md

🧪 Output Examples
Positive Cloud	Negative Cloud

	
⚙️ Setup

Clone the project:

git clone https://github.com/kuohsinyu/Hiproject
cd Hiproject


Create virtual environment:

python3 -m venv .venv
source .venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Set your API key:

export YOUTUBE_API_KEY="your_api_key"


Run:

python hiproject.py

🌐 Azure Integration (Future Work)

This project can be expanded into a cloud-native solution:

Component	Azure Service
Comment ingestion	Azure Functions
Storage	Azure Blob Storage
Sentiment scoring	Azure Cognitive Services – Text Analytics
Dashboard	Power BI or Azure Static Web App

This aligns directly with Microsoft Cloud & AI Solution Engineer workflows.

🚧 Future Improvements

Interactive web dashboard for real-time visualization

Multi-language support

Topic clustering

Deployment as an API endpoint

Full Azure implementation

👤 Author

Hsin-Yu Kuo (郭昕宇)
Exchange Student @ Heidelberg University
Learning Technology • NLP • AI Prototyping
GitHub: https://github.com/kuohsinyu
