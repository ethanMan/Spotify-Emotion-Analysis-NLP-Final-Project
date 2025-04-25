import re

def clean_lyrics(text):
    """
    Cleans the lyrics by removing unwanted characters and formatting.
    Args:
        text (str): The lyrics to clean.
    Returns:
        str: The cleaned lyrics.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()
