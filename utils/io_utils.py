import pandas as pd


def load_data(file_path) -> pd.DataFrame:
    """
    Load data from a CSV file into a DataFrame.
    """
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    return df


def save_data(df, file_path) -> None:
    """
    Save a DataFrame to a CSV file.
    """
    df.to_csv(file_path, index=False)
    print(f"Saved dataframe to {file_path}")


def load_txt(file_path) -> pd.DataFrame:
    """
    Load data from a tab-separated text file into a DataFrame.
    """
    df = pd.read_csv(file_path, sep="\t", header=None, names=("review", "label"))
    print(f"Loaded {len(df)} rows from {file_path}")
    return df


def fetch_lyrics(nrows=100000) -> pd.DataFrame:
    """
    This function fetches the lyrics from a CSV file and filters them based on certain criteria.
    It returns a DataFrame containing the filtered lyrics.
    """
    # Read the CSV file
    df = pd.read_csv("data/datasets/song_lyrics.csv", nrows=nrows)

    # Filter the DataFrame
    df = df[df["lyrics"].notna()]
    df = df[df["language"] == "en"]
    df = df[df["views"] > 10000]

    # Select relevant columns
    df = df[["title", "tag", "views", "artist", "lyrics"]]

    return df
