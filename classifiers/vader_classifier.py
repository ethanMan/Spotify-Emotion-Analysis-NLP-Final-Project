from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from utils.text_utils import clean_lyrics

class VaderClassifier:
    """
    A class to analyze the sentiment of song lyrics using VADER (Valence Aware Dictionary and sEntiment Reasoner).
    """
    def __init__(self, cleaner=clean_lyrics):
        self.analyzer = SentimentIntensityAnalyzer()
        self.cleaner = cleaner

    def get_sentiment_category(self, score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    def analyze_lyrics(self, lyrics):
        if not isinstance(lyrics, str) or not lyrics.strip():
            return {"compound": 0.0, "neg": 0.0, "neu": 1.0, "pos": 0.0, "sentiment": "neutral"}
        try:
            cleaned = self.cleaner(lyrics)
            scores = self.analyzer.polarity_scores(cleaned)
            scores["sentiment"] = self.get_sentiment_category(scores["compound"])
            return scores
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {"compound": 0.0, "neg": 0.0, "neu": 1.0, "pos": 0.0, "sentiment": "neutral"}

    def analyze_dataframe(self, df, text_column="lyrics"):
        tqdm.pandas(desc="Analyzing sentiment")
        sentiment_results = df[text_column].progress_apply(self.analyze_lyrics)

        df["sentiment_compound"] = sentiment_results.apply(lambda x: x["compound"])
        df["sentiment_negative"] = sentiment_results.apply(lambda x: x["neg"])
        df["sentiment_neutral"] = sentiment_results.apply(lambda x: x["neu"])
        df["sentiment_positive"] = sentiment_results.apply(lambda x: x["pos"])
        df["sentiment_category"] = sentiment_results.apply(lambda x: x["sentiment"])

        return df
