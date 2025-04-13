import re
import ast
import joblib
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class EmotionClassifier:
    """
    A class to classify emotions from text using a logistic regression model.
    """
    def __init__(self):
        self.emotions = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", 
            "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", 
            "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", 
            "remorse", "sadness", "surprise", "neutral"
        ]
        self.emotion_mapping = {
            2: 2, 3: 2, 10: 2,
            11: 11,
            0: 17, 1: 17, 4: 17, 5: 17, 8: 17, 13: 17, 15: 17, 17: 17, 18: 17, 20: 17, 21: 17, 23: 17,
            14: 14, 19: 14,
            9: 25, 12: 25, 16: 25, 24: 25, 25: 25,
            6: 26, 7: 26, 22: 26, 26: 26
        }
        self.model = None
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text):
        """Preprocess the text by converting to lowercase, removing punctuation, and lemmatizing."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        words = word_tokenize(text)
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return " ".join(words)
    
    def process_labels(self, label_str):
        try:
            label_obj = ast.literal_eval(label_str)
            if isinstance(label_obj, tuple):
                return list(label_obj)
            elif isinstance(label_obj, int):
                return [label_obj]
            else:
                return [label_obj]
        except (ValueError, SyntaxError):
            return [int(x) for x in label_str.split(',')]

    def load_and_prepare_data(self, filepath):
        df = pd.read_csv(filepath, delimiter="\t", names=["text", "label", "id"])
        df["text"] = df["text"].apply(self.preprocess_text)
        df["label"] = df["label"].apply(self.process_labels)
        df = df.explode("label")
        df["label"] = df["label"].astype(int)
        df = df[df["label"] != 27]  # Remove neutral
        df["label"] = df["label"].map(self.emotion_mapping)
        return df.dropna()

    def evaluate_with_cross_validation(self, X, y, cv=5):
        pipeline = make_pipeline(
            TfidfVectorizer(min_df=5, max_df=0.8),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, solver='liblinear')
        )
        scoring = {
            'accuracy': 'accuracy',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro',
            'f1_macro': 'f1_macro'
        }
        results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
        for metric in scoring:
            mean = results[f'test_{metric}'].mean()
            std = results[f'test_{metric}'].std()
            print(f"{metric.capitalize()}: {mean:.4f} ± {std:.4f}")
        return results

    def tune_hyperparameters(self, X, y):
        pipeline = make_pipeline(
            TfidfVectorizer(),
            LogisticRegression(class_weight="balanced", max_iter=1000)
        )
        param_grid = {
            'tfidfvectorizer__min_df': [2, 5],
            'tfidfvectorizer__max_df': [0.8, 0.9],
            'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
            'logisticregression__C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'logisticregression__solver': ['liblinear', 'saga']
        }
        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1_macro", verbose=1, n_jobs=-1)
        grid.fit(X, y)
        print("Best parameters:")
        for param, val in grid.best_params_.items():
            print(f"  {param}: {val}")
        print(f"Best F1 Score: {grid.best_score_:.4f}")
        self.model = grid.best_estimator_

    def train_final_model(self, X, y):
        # If hyperparameter tuning was done, the model is already set
        if self.model is None:
            # Fallback to default parameters if no tuning was done
            self.model = make_pipeline(
                TfidfVectorizer(min_df=5, max_df=0.8, ngram_range=(1, 2)),
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, solver='liblinear')
            )
        
        # Train the model (if it was set by tune_hyperparameters, we're just retraining on full data)
        self.model.fit(X, y)

    def save_model(self, filename="models/emotion_classifier_model.joblib"):
        joblib.dump(self.model, filename)
        joblib.dump({
            'model': self.model,
            'emotions': self.emotions,
            'emotion_mapping': self.emotion_mapping
        }, "models/emotion_classifier_with_info.joblib")
        print(f"Model saved to {filename} and emotion_classifier_with_info.joblib")

    def predict(self, text, model_info=None):
        if model_info is None:
            model_info = joblib.load("models.emotion_classifier_with_info.joblib")
        model = model_info['model']
        emotions_list = model_info['emotions']
        processed = self.preprocess_text(text)
        proba = model.predict_proba([processed])[0]
        prediction = model.predict([processed])[0]
        classes = model.classes_
        pred_idx = np.where(classes == prediction)[0][0]
        top_idxs = proba.argsort()[-3:][::-1]
        top_emotions = [
            {
                'emotion': emotions_list[classes[i]] if classes[i] < len(emotions_list) else f"Unknown ({classes[i]})",
                'probability': proba[i]
            }
            for i in top_idxs
        ]
        return {
            'emotion': emotions_list[prediction] if prediction < len(emotions_list) else f"Unknown ({prediction})",
            'confidence': proba[pred_idx],
            'top_emotions': top_emotions
        }

if __name__ == "__main__":

    # Train model
    clf = EmotionClassifier()
    df_train = clf.load_and_prepare_data("data/datasets/goemotions/train.tsv")
    df_test = clf.load_and_prepare_data("data/datasets/goemotions/test.tsv")
    X_train = df_train["text"]
    y_train = df_train["label"]
    X_test = df_test["text"]
    y_test = df_test["label"]

    clf.evaluate_with_cross_validation(X_train, y_train)
    clf.tune_hyperparameters(X_train, y_train)
    clf.train_final_model(X_train, y_train)
    clf.save_model()

    # Example prediction
    sample = "I am really excited about this new project!"
    prediction = clf.predict(sample)
    print(f"\nSample: '{sample}'")
    print(f"Predicted Emotion: {prediction['emotion']} (Confidence: {prediction['confidence']:.2f})")
    print("Top Emotions:")
    for e in prediction['top_emotions']:
        print(f"  {e['emotion']}: {e['probability']:.2f}")

'''
### Usage Instructions ###

# 1. Training a new model:
clf = EmotionClassifier()
df_train = clf.load_and_prepare_data("path/to/train.tsv")
df_test = clf.load_and_prepare_data("path/to/test.tsv")
X_train = df_train["text"]
y_train = df_train["label"]

# Optional: Evaluate with cross-validation
clf.evaluate_with_cross_validation(X_train, y_train)

# Optional: Tune hyperparameters
clf.tune_hyperparameters(X_train, y_train)

# Train final model
clf.train_final_model(X_train, y_train)
clf.save_model("models/your_model_name.joblib")

# 2. Using a pre-trained model for prediction:
clf = EmotionClassifier()
model_info = joblib.load("models/emotion_classifier_with_info.joblib")
text = "I'm so excited about this project!"
prediction = clf.predict(text, model_info)
print(f"Emotion: {prediction['emotion']} (Confidence: {prediction['confidence']:.2f})")
print("Top emotions:")
for emotion in prediction['top_emotions']:
    print(f"  {emotion['emotion']}: {emotion['probability']:.2f}")
'''