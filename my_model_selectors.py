import math
import statistics
import warnings
import sys

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        pass

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        models = dict()

        for states_number in range(self.min_n_components, self.max_n_components + 1):
            model = self.base_model(states_number)
            try:
                logL = model.score(self.X, self.lengths)
                logN = np.log(len(self.X))
                # Number of parameters is the sum of following items:
                #   - Initial state probabilities = number of states
                #   - Transition probabilities = number of states * (number of states - 1)
                #   - Emission probabilities = number of states * number of features * 2
                p = states_number + states_number * (states_number - 1) + \
                    states_number * model.n_features * 2
                bic = -2 * logL + p * logN
            except Exception as e:
                bic = float("+inf")

            models[model] = bic

        best_model = min(models, key=models.get)

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        models = dict()
        for states_number in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(states_number)

                dic = model.score(self.X, self.lengths)

                scores = [model.score(X, lengths) for word, (X, lengths) in self.hwords.items() if
                          word != self.this_word]
                dic -= np.mean(scores)

                models[model] = dic
            except:
                models[model] = float("-inf")

        best_model = max(models, key=models.get)

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        models = dict()
        for states_number in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(states_number)

                split_method = KFold(n_splits=2)
                scores = list()
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                    train_model = self.base_model(states_number)
                    X, lengths = combine_sequences(cv_test_idx, self.sequences)
                    score = train_model.score(X, lengths)
                    scores.append(score)

                models[model] = np.mean(scores)
            except:
                models[model] = float("-inf")

        best_model = max(models, key=models.get)

        return best_model
