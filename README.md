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
- **Emotion Detection:** [DistilBERT](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base#emotion-english-distilroberta-base)
- **Sentiment Analysis:** [VADER](https://github.com/cjhutto/vaderSentiment) or **Naive Bayes Classifier**
- **Input Analysis:** Logistic Regression

---

## 🚀 Quick Start

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/Spotify-Emotion-Analysis-NLP-Final-Project.git
    cd Spotify-Emotion-Analysis-NLP-Final-Project
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

## 📂 Datasets

### 🎵 `song_lyrics.csv` (Kaggle)
- **Source:** Scraped from Genius (via [Kaggle](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information?resource=download)).
- **Size:** 3,093,218 songs (~9 GB)
- **Key Fields:** `title`, `tag` (genre), `artist`, `views`, `lyrics`, `language`
- **Note:** Omitted from submission due to size; a processed subset (`filtered_lyrics.csv`) was used.
- **Filtered Criteria:**  
  - English language  
  - Has lyrics  
  - >10,000 views
- **Filtered Size:** 2,000,000 songs
- **Preprocessing:** Handled via `fetch_lyrics` in `io_utils.py`.

### 📱 `amazon_cells_labeled.txt` (Kaggle)
- **Source:** [Kaggle Amazon Cell Phones Dataset](https://www.kaggle.com/datasets/marklvl/sentiment-labelled-sentences-data-set).
- **Size:** 1,000 Amazon product reviews (cell phones & accessories)
- **Labels:** Binary Sentiment
  - 1 = Positive
  - 0 = Negative
- **Scoring Rule:**
  - Scores 4–5 → Positive
  - Scores 1–2 → Negative

### 💬 GoEmotions (`train.tsv` and `test.tsv`) (GitHub)
- **Source:** [Google Research GitHub - GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
- **Size:** 58,000 Reddit comments
  - Training set: 43,410
  - Testing set: 5,427
- **Labels:** 27 Emotion Categories + Neutral
- **Emotion Categories:**  
  admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise

---

## ✨ Future Improvements
- Fine-tune emotion detection models for more nuanced emotions.
- Expand the song database for better playlist diversity.
- Build a front-end interface for easier user interaction.

---

## 📬 Contact
Questions or suggestions? Feel free to reach out!

