from transformers import pipeline
from tqdm import tqdm
from utils.text_utils import clean_lyrics

class DistilBertClassifier:
    """
    A class to classify song lyrics into emotional categories using a DistilBERT model.
    """
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base", cleaner=clean_lyrics):
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=1,
            truncation=True
        )
        self.cleaner = cleaner

    def classify(self, lyrics):
        if not isinstance(lyrics, str) or not lyrics.strip():
            return "neutral"
        try:
            cleaned = self.cleaner(lyrics)
            result = self.classifier(cleaned)
            # Handle both possible output structures
            if isinstance(result, list) and isinstance(result[0], dict):
                return result[0]["label"].lower()
            elif isinstance(result[0], list) and isinstance(result[0][0], dict):
                return result[0][0]["label"].lower()
            return "neutral"
        except Exception as e:
            print(f"Error classifying lyrics: {e}")
            return "neutral"

    def classify_dataframe(self, df, text_column="lyrics", output_column="mood"):
        tqdm.pandas(desc="Classifying lyrics")
        df[output_column] = df[text_column].progress_apply(self.classify)
        return df
