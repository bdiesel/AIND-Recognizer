import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """

    probabilities = []
    guesses = []

    # Implement the recognizer
    test_words = sorted(test_set.get_all_sequences().keys())

    for test_word in test_words:
        X, lengths = test_set.get_item_Xlengths(test_word)
        probs_dict = {}
        best_score = float("-inf")
        best_guess = ""

        for train_word, model in models.items():
            try:
                log_prob = model.score(X, lengths)
            except:
                log_prob = float("-inf")
            probs_dict[train_word] = log_prob
            if log_prob > best_score:
                best_score = log_prob
                best_guess = train_word

        guesses.append(best_guess)
        probabilities.append(probs_dict)

    return probabilities, guesses