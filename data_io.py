import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    return df

def save_data(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"Saved dataframe to {file_path}")

def load_txt(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None, names=('review', 'label'))
    print(f"Loaded {len(df)} rows from {file_path}")
    return df
    