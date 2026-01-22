# Intent Assist – Hybrid AI Intent Detection Chatbot

An AI-powered chatbot that detects user intent using a hybrid approach combining 
machine learning (TF-IDF + Naive Bayes) with rule-based safety overrides.

Features:
- Multi-intent classification (greeting, query, weather, jokes, boredom, etc.)
- Safety-critical intent detection (danger, self-harm, harassment, stalking)
- Text normalization to handle elongated words (e.g., "hellooo" → "hello")
- Hybrid rule + ML decision pipeline with confidence scoring
- Automatic helpline and safety recommendations for high-risk intents
- Deployed using Streamlit

Tech Stack:
- Python
- scikit-learn
- pandas, numpy
- Streamlit
- Joblib
- NLP (TF-IDF, n-grams)

This project demonstrates hybrid NLP system design and safety-aware AI deployment.
