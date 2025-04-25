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