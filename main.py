from config import CLASSIFIED_FILE, SENTIMENT_FILE
from data_io import load_data, save_data, load_txt
from DistilBertClassifier import DistilBertClassifier
from VaderClassifier import VaderClassifier
from NaiveBayesClassifier import NaiveBayesClassifier
from input_analyzer import EmotionClassifier
import joblib


def classify_lyrics(df):
    # Emotion classification
    distilBertClassifier = DistilBertClassifier()
    df = distilBertClassifier.classify_dataframe(df)
    print("Mood distribution:\n", df["mood"].value_counts())
    save_data(df, CLASSIFIED_FILE)

    # Sentiment analysis
    vaderClassifier = VaderClassifier()
    df = vaderClassifier.analyze_dataframe(df)
    print("Sentiment category distribution:\n", df["sentiment_category"].value_counts())
    print("Compound score stats:\n", df["sentiment_compound"].describe())
    save_data(df, SENTIMENT_FILE)

    df = load_data(SENTIMENT_FILE)
    # Sentiment analysis
    nb_classifier = NaiveBayesClassifier()
    # Load IMDB dataset
    imdb_df = load_txt(
        "data/datasets/sentiment labelled sentences/amazon_cells_labelled.txt"
    )

    # Train the classifier
    nb_classifier.train(imdb_df["review"].tolist(), imdb_df["label"].tolist())
    nb_classifier.analyze_sentiment_in_dataframe(df)
    print(
        "Naive Bayes sentiment category distribution:\n",
        df["sentiment_category_nb"].value_counts(),
    )
    save_data(df, SENTIMENT_FILE)
    return df


def calculate_accuracy(df):
    # Calculate accuracy
    correct_count = 0
    total = 0
    for row in df.itertuples():
        if row.sentiment_category == "positive" and row.sentiment_category_nb == 1:
            correct_count += 1
        elif row.sentiment_category == "negative" and row.sentiment_category_nb == 0:
            correct_count += 1
        if row.sentiment_category != "neutral":
            total += 1
    accuracy = correct_count / total if total > 0 else 0
    print(f"Accuracy of Naive Bayes classifier: {accuracy:.2%}")


def generate_playlists_emotion(mood, num_songs=10):
    df = load_data(CLASSIFIED_FILE)
    # Filter the DataFrame based on the mood
    filtered_df = df[df["mood"] == mood]
    # Sort by number of views
    sorted_df = filtered_df.sort_values(by="views", ascending=False)
    # Select the top songs
    top_songs = sorted_df.head(num_songs)
    print(f"Top songs for mood '{mood}':")
    print(top_songs[["title", "artist", "views"]])


def generate_playlists_sentiment_vader(sentiment, num_songs=10):
    df = load_data(SENTIMENT_FILE)
    # Filter based on sentiment
    filtered_df = df[df["sentiment_category"] == sentiment]
    # Sort by views
    sorted_df = filtered_df.sort_values(by="views", ascending=False)
    # Select the top songs
    top_songs = sorted_df.head(num_songs)
    print(f"Top songs for sentiment '{sentiment}':")
    print(top_songs[["title", "artist", "views"]])


def generate_playlists_sentiment_nb(sentiment, num_songs=10):
    df = load_data(SENTIMENT_FILE)
    # Filter based on sentiment
    filtered_df = df[df["sentiment_category_nb"] == sentiment]
    # Sort by views
    sorted_df = filtered_df.sort_values(by="views", ascending=False)
    # Select the top songs
    top_songs = sorted_df.head(num_songs)
    print(f"Top songs for sentiment '{sentiment}':")
    print(top_songs[["title", "artist", "views"]])


def main():
    # generate_playlists_emotion("joy")
    # generate_playlists_emotion("anger")
    # generate_playlists_emotion("fear")
    # generate_playlists_emotion("sadness")
    # generate_playlists_emotion("surprise")
    # generate_playlists_emotion("disgust")

    generate_playlists_sentiment_vader("positive", 20)
    generate_playlists_sentiment_vader("negative", 20)

    generate_playlists_sentiment_nb(1, 20)
    generate_playlists_sentiment_nb(0, 20)


    classifier = EmotionClassifier()
    model_info = joblib.load("emotion_classifier_with_info.joblib")
    text = input("Enter how you're feeling: ")
    result = classifier.predict(text, model_info=model_info)
    emotion = result["emotion"]

    print(f"Emotion: {result['emotion']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print("Top emotions:")
    for e in result["top_emotions"]:
        print(f"  {e['emotion']}: {e['probability']:.2f}")

    generate_playlists_emotion(emotion)


if __name__ == "__main__":
    main()
