from utils.config import DATA_FILE, ANALYZED_FILE, CLASSIFIED_FILE, SENTIMENT_FILE, EMOTION_MODEL_PATH_INFO
from utils.io_utils import load_data
from analysis.classify import classify_emotions, analyze_sentiment_vader, analyze_sentiment_nb
from analysis.accuracy import calculate_accuracy
from classifiers.lr_classifier import EmotionClassifier
from playlist.generator import generate_playlist
from joblib import load

"""
What you can do with this code:
This code is designed to analyze and classify emotions in song lyrics, perform sentiment analysis, and generate playlists based on the results. Below are the main functionalities:
1. Loading data:
    - Use the `load_data` function to load the dataset from the specified path.
    - The dataset should be in CSV format and contain lyrics data.
    - A path to an unanalyzed dataset is provided in the `DATA_FILE` variable.
    - Paths to analyzed datasets are provided in the `ANALYZED_FILE`, `CLASSIFIED_FILE`, and `SENTIMENT_FILE` variables.
    - `ANALYZED_FILE` is the path to the dataset after emotion classification and sentiment analysis.
    - `CLASSIFIED_FILE` is the path to the dataset after emotion classification.
    - `SENTIMENT_FILE` is the path to the dataset after sentiment analysis.
    - Example: df = load_data(DATA_FILE)
2. Classifying emotions:
    - Use the `classify_emotions` function to classify the emotions in the dataset.
    - The function should take a DataFrame as input and return a DataFrame with classified emotions.
    - Example: df = classify_emotions(df)
3. Analyzing sentiment with VADER:
    - Use the `analyze_sentiment_vader` function to analyze the sentiment of the dataset using VADER.
    - The function should take a DataFrame as input and return a DataFrame with analyzed sentiment.
    - Example: df = analyze_sentiment_vader(df)
4. Analyzing sentiment with Naive Bayes:
    - Use the `analyze_sentiment_nb` function to analyze the sentiment of the dataset using Naive Bayes.
    - The function should take a DataFrame as input and return a DataFrame with analyzed sentiment.
    - Example: df = analyze_sentiment_nb(df)
5. Generating playlists:
    - Use the `generate_playlist` function to generate playlists based on classified emotions or analyzed sentiment.
    - The function should take a DataFrame, a column name, a value to filter by, and the number of songs to return.
    - Example (emotion): generate_playlist(df, "mood", "happy", 10)
    - Example (sentiment): generate_playlist(df, "sentiment_category", "positive", 10)
    - Example (Naive Bayes): generate_playlist(df, "sentiment_category_nb", 1, 10)
6. Calculating accuracy:
    - Use the `calculate_accuracy` function to calculate the accuracy of the sentiment analysis (compares VADER and Naive Bayes implementation).
    - The function should take a sentiment-analyzed DataFrame as input and return the accuracy.
    - Example: accuracy = calculate_accuracy(df)
7. Emotion prediction:
    - Use the `EmotionClassifier` class to predict the emotion of a given text.
    - The class should be initialized with a model path.
    - Use the `predict` method to get the predicted emotion and its confidence.
    - Example: classifier = EmotionClassifier(EMOTION_MODEL_PATH); result = classifier.predict(text)
8. User input to corresponding playlist:
    - Use the `prompt_user_for_emotion` function to prompt the user for their current emotion.
    - The function should take user input and return the corresponding playlist based on the predicted emotion.
    - Example: playlist = prompt_user_for_emotion()
"""

def prompt_user_for_emotion():
    text = input("Enter how you're feeling: ")
    classifier = EmotionClassifier()
    model_info = load("models/emotion_classifier_with_info.joblib")
    result = classifier.predict(text, model_info=model_info)

    print(f"\nEmotion: {result['emotion']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print("Top emotions:")
    for e in result["top_emotions"]:
        print(f"  {e['emotion']}: {e['probability']:.2f}")

    return result["emotion"]

def main():
    # df = load_data(DATA_FILE)
    # classify_emotions(df)
    # df = load_data(ANALYZED_FILE)
    # analyze_sentiment_vader(df)
    # df = load_data(ANALYZED_FILE)
    # analyze_sentiment_nb(df)

    df = load_data(CLASSIFIED_FILE)
    moods = ["joy", "anger", "fear", "sadness", "surprise", "disgust"]
    for mood in moods:
        print(f"Top songs for mood '{mood}':")
        print(generate_playlist(df, "mood", mood, 10))
        print()

    df = load_data(SENTIMENT_FILE)
    print("Top songs for sentiment 'positive':")
    print(generate_playlist(df, "sentiment_category", "positive", 10))
    print()
    print("Top songs for sentiment 'negative':")
    print(generate_playlist(df, "sentiment_category", "negative", 10))
    print()

    print(f"Top songs for sentiment '{1}':")
    print(generate_playlist(df, "sentiment_category_nb", 1, 10))
    print()
    print(f"Top songs for sentiment '{0}':")
    print(generate_playlist(df, "sentiment_category_nb", 0, 10))
    
    df = load_data(CLASSIFIED_FILE)
    emotion = prompt_user_for_emotion()
    print(generate_playlist(df, "mood", emotion, 10))


if __name__ == "__main__":
    main()
