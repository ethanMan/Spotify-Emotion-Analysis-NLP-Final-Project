import os

BASE_DIR = "/Users/Superfcbear/Library/Mobile Documents/com~apple~CloudDocs/[current] CS4120/Final Project/"
DATA_FILE = os.path.join(BASE_DIR, "data/filtered_lyrics.csv")
CLASSIFIED_FILE = os.path.join(BASE_DIR, "data/classified_lyrics.csv")
SENTIMENT_FILE = os.path.join(BASE_DIR, "data/sentiment_analyzed_lyrics.csv")
ANALYZED_FILE = os.path.join(BASE_DIR, "data/analyzed_lyrics.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")
EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_classifier_model.joblib")
EMOTION_MODEL_PATH_INFO = os.path.join(MODELS_DIR, "emotion_classifier_with_info.joblib")
