import os
import pandas as pd

os.chdir('/Users/Superfcbear/Library/Mobile Documents/com~apple~CloudDocs/[current] CS4120/Final Project/')

def fetch_lyrics(nrows=100000):
    """
    This function fetches the lyrics from a CSV file and filters them based on certain criteria.
    It returns a DataFrame containing the filtered lyrics.
    """
    # Read the CSV file
    df = pd.read_csv('data/datasets/song_lyrics.csv', nrows=nrows)
    
    # Filter the DataFrame
    df = df[df['lyrics'].notna()]
    df = df[df['language'] == 'en']
    df = df[df['views'] > 10000]
    
    # Select relevant columns
    df = df[['title', 'tag', 'views', 'artist', 'lyrics']]
    
    return df

if __name__ == "__main__":
    # Fetch the lyrics
    # lyrics_df = fetch_lyrics(nrows=2000000)
    
    # Save the DataFrame to a CSV file
    # lyrics_df.to_csv('filtered_lyrics.csv', index=False)
    data = pd.read_csv('data/filtered_lyrics.csv')
    
    # Print the shape of the DataFrame
    print(data.shape)

    # Print the breakdown of the csv by genre
    genre_counts = data['tag'].value_counts()
    print(genre_counts)

    artist_counts = data['artist'].value_counts()
    print(artist_counts)