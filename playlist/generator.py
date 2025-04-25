def generate_playlist(df, filter_column, filter_value, num_songs=10):
    """
    Generate a playlist based on a filter column and value.

    Parameters:
    - df: DataFrame containing song data
    - filter_column: Column to filter by (e.g., 'mood', 'sentiment_category', 'sentiment_category_nb')
    - filter_value: Value to filter by (e.g., 'joy', 'positive', 1)
    - num_songs: Number of songs to return
    """
    filtered_df = df[df[filter_column] == filter_value]
    sorted_df = filtered_df.sort_values(by="views", ascending=False)
    return sorted_df.head(num_songs)[["title", "artist", "views"]]
