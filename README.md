# 🎵 Spotify Emotion/Sentiment Playlist Generator

Generate a personalized 50-song Spotify playlist based on the emotion or sentiment of any input sentence!

This project was built as the Final Project for **CS4120 (Natural Language Processing)** at **Northeastern University**.

[📄 View Full Project Report](https://docs.google.com/document/d/13dphRA2-PVXfCEFzkE9DUUFabCtXcmKODQrxNFChL6w/edit?usp=sharing)

---

## 🧠 How It Works
1. User inputs a sentence.
2. A **Logistic Regression** model predicts the sentence’s **emotion** or **sentiment**.
3. Based on the prediction, a playlist is curated from pre-classified songs.

**Models Used:**
- **Emotion Detection:** [DistilBERT](https://huggingface.co/distilbert-base-uncased)
- **Sentiment Analysis:** [VADER](https://github.com/cjhutto/vaderSentiment) or **Naive Bayes Classifier**
- **Input Analysis:** Logistic Regression

---

## 🚀 Quick Start

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/spotify-emotion-playlist.git
    cd spotify-emotion-playlist
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    python main.py
    ```

*Detailed instructions are available inside* `main.py`.

---

## 📚 Technologies Used
- Python
- Scikit-learn
- Hugging Face Transformers (DistilBERT)
- NLTK (VADER Sentiment Analyzer)
- Spotify API

---

## ✨ Future Improvements
- Fine-tune emotion detection models for more nuanced emotions.
- Expand the song database for better playlist diversity.
- Build a front-end interface for easier user interaction.

---

## 📬 Contact
Questions or suggestions? Feel free to reach out!

