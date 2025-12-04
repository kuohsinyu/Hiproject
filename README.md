🎧 YouTube Chinese Sentiment Analyzer

AI-driven NLP pipeline for understanding polarized Chinese discussions

<p align="center"> <img src="positive_wordcloud.png" width="40%" /> <img src="negative_wordcloud.png" width="40%" /> </p>
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

YouTube Chinese Sentiment Analyzer

An end-to-end Python NLP pipeline that:

collects YouTube comments via API

performs Chinese sentiment analysis (1–5 levels, positive vs negative)

visualizes results as Taiwan-shaped word clouds

This project identifies positive vs. negative opinion clusters across highly polarized Chinese-language videos (politics, social issues, entertainment, etc.).

| Negative Cloud <img width="357" height="636" alt="image" src="https://github.com/user-attachments/assets/096c0f39-c771-4d54-a070-0afde21b9d22" />
| Positive Cloud <img width="357" height="636" alt="image" src="https://github.com/user-attachments/assets/2c103a63-10e7-4a1e-8b5f-e7df3d46f013" />

🚀 Features

Fetch top-level comments and replies from any YouTube video

Chinese tokenization + keyword extraction using jieba

Fine-tuned BERT (6-level sentiment classification)

Automatically group comments into positive and negative clusters

Generate Taiwan-shaped word clouds to show dominant themes

Summarize thousands of comments into a glanceable visualization

🧠 Tech Stack

Python, PyTorch, Transformers (BERT)

YouTube Data API v3

jieba, wordcloud, matplotlib, Pillow

pandas / numpy

JSON dataset (trainword.json, excluded for privacy)

📁 Project Structure
Hiproject/
│── hiproject.py          # Main script, end-to-end pipeline
│── extract.py            # (optional) additional data processing
│── trained_model/        # Fine-tuned BERT model (excluded from repo)
│── taiwan.jpg            # Mask image for word clouds
│── requirements.txt      # Required packages
│── README.md

🧩 How It Works (Pipeline)

User enters a YouTube video URL

System fetches all comments + replies via YouTube API

Text preprocessing → Chinese character extraction → jieba tokenization

BERT model assigns sentiment labels (0–5)

Comments grouped into positive (labels 1,5) and negative (2,3,4)

Word frequencies extracted

Generate:

positive_wordcloud.png

negative_wordcloud.png

🔧 Setup

1. Clone repository
git clone https://github.com/kuohsinyu/Hiproject
cd Hiproject

2. Create virtual environment (optional)
python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Set YouTube API key
export YOUTUBE_API_KEY="your_api_key_here"

5. Run
python hiproject.py
