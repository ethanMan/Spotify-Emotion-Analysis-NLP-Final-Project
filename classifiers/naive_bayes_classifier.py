import numpy as np
from collections import Counter
import nltk
from tqdm import tqdm

def tokenize(sentence:str)->list:
    """
    This function tokenizes a sentence using nltk.
    Args:
        sentence (str): The sentence to be tokenized.
    Returns:
        list: A list of tokens.
    """
    tokenized_sentence = nltk.word_tokenize(sentence)
    return tokenized_sentence


class NaiveBayesClassifier:
    """ 
    This class creates an instance of the Naive Bayes classifier used to classify the sentiment of song lyrics.
    """
    def __init__(self):
        self.classes = set()
        self.vocab = set()
        self.word_counts = {}
        self.word_probabilities = {}
        self.class_counts = Counter()

    # PARTIALLY PROVIDED
    # Students: you can ignore or use/update this function as you see fit
    def init_counter(self, X:list, y:list)->None:
        """
        This function initializes the vocabulary of the dataset. 
        It also creates a bag-of-words representation for each genre. 
        This can be used to retrieve the count of a word given a class.
        Args:
            X (list): A list of lyrics as strings (not tokenized).
            y (list): A list of target labels corresponding to the lyrics.
        """   
        self.classes = set(y)
        self.word_counts = {label:Counter() for label in self.classes}
        self.word_probabilities = {label:{} for label in self.classes}
        self.prior_probs = {}

        for sentence, label in zip(X,y):
            tokenized_sentence = tokenize(sentence)
            self.vocab.update(tokenized_sentence)
            self.word_counts[label].update(tokenized_sentence)
            self.class_counts[label] += 1
            

    def train(self, X:list, y:list)->None:
        """
        Computes the prior probabilities of each class and the likelihood probabilities of each word in each class,
        and stores them within object variables.
        These will then be used while predicting the label of lyrics for a song.

        Args:
            X (list): A list of sentences.
            y (list): A list of target labels corresponding to the sentences.
        """
        N_total = len(X)
        self.init_counter(X, y)

        self.prior_probs = {label: self.class_counts[label] / N_total for label in self.classes}
        for label in self.classes:
            for word in self.vocab:
                self.word_probabilities[label][word] = self.get_likelihood(word, label)

    
    def get_vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.
        Returns:
            int: The size of the vocabulary.
        """
        return len(self.vocab)

    
    def get_classes(self) -> set:
        """
        Returns the set of classes.
        Returns:
            set: The set of classes.
        """
        return self.classes

    
    def get_prior(self, label:str) -> float:
        """
        Returns the prior probability of a class (label).
        Args:
            label (str): The label for which the prior probability is to be returned.
        Returns:
            float: The prior probability of the label.
        """
        return self.prior_probs[label]

    
    def get_likelihood(self, word:str, label:str) -> float:
        """
        Computes the probability of a word given a label.
        That is, this function returns P(word|label).

        Always use Laplace smoothing while computing the probability.

        Args:
            word (str): The word for which the likelihood is to be returned.
            label (str): The label for which the likelihood is to be returned.
        Returns:
            float: The likelihood of the word given the label.
        """
        word_count = self.word_counts[label][word] + 1
        total_words = sum(self.word_counts[label].values()) + self.get_vocab_size()

        return word_count / total_words

    
    def predict(self, X: str)-> str:
        """
        Takes in a movie plot.
        Loops over the classes (labels) and computes the probability that the movie belongs to each one.  
        Returns the most likely class.

        Args:
            X (str): A string representing lyrics of a song.

        Returns:
            str: The label of the lyrics (label).
        """ 
        # Initialize variables to track the best class
        max_log_prob = float('-inf')
        prediction = None

        # Tokenize the input text
        tokens = tokenize(X)
        
        # Calculate probability for each class
        for label in self.classes:
            # Start with log of prior probability
            log_prob = np.log(self.get_prior(label))
            
            # Add log of likelihood for each word
            for word in tokens:
                if word in self.vocab:
                    # Use pre-computed word probabilities if available
                    if word in self.word_probabilities[label]:
                        log_prob += np.log(self.word_probabilities[label][word])
                    else:
                        # Calculate on the fly if needed
                        log_prob += np.log(self.get_likelihood(word, label))
                # Add a small penalty for unknown words
                log_prob += np.log(1.0 / (sum(self.word_counts[label].values()) + self.get_vocab_size()))
            
            # Update prediction if this class has higher probability
            if log_prob > max_log_prob:
                max_log_prob = log_prob
                prediction = label

        return prediction
    
    def analyze_sentiment_in_dataframe(self, df):
        tqdm.pandas(desc="Analyzing sentiment")
        sentiment_results = df["lyrics"].progress_apply(self.predict)
        df["sentiment_category_nb"] = sentiment_results
        return df
        
