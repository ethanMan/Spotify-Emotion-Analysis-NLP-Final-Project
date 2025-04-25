from utils.config import CLASSIFIED_FILE, SENTIMENT_FILE, ANALYZED_FILE
from utils.io_utils import save_data, load_txt
from classifiers.distilbert_classifier import DistilBertClassifier
from classifiers.vader_classifier import VaderClassifier
from classifiers.naive_bayes_classifier import NaiveBayesClassifier

def classify_emotions(df, output_path=ANALYZED_FILE):
    classifier = DistilBertClassifier()
    df = classifier.classify_dataframe(df)
    save_data(df, output_path)
    return df

def analyze_sentiment_vader(df, output_path=ANALYZED_FILE):
    classifier = VaderClassifier()
    df = classifier.analyze_dataframe(df)
    save_data(df, SENTIMENT_FILE)
    return df

def analyze_sentiment_nb(df, dataset_path, output_path=ANALYZED_FILE):
    classifier = NaiveBayesClassifier()
    dataset = load_txt(dataset_path)
    classifier.train(dataset["review"].tolist(), dataset["label"].tolist())
    df = classifier.analyze_sentiment_in_dataframe(df)
    save_data(df, output_path)
    return df