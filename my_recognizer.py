import warnings
from asl_data import SinglesData, AslDb


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
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses

    for X, lengths in test_set.get_all_Xlengths().values():
        words_score = dict()
        for word, model in models.items():
            try:
                score = model.score(X, lengths)
            except Exception as e:
                score = float("-inf")
            words_score[word] = score

        recognized_word = max(words_score, key=words_score.get)
        guesses.append(recognized_word)
        probabilities.append(words_score)

    return probabilities, guesses

from asl_utils import show_errors
from my_model_selectors_advanced import *
warnings.filterwarnings('ignore')

features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']
features_custom = ['hand_distance']

asl = AslDb()

def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word,
                        n_constant=3).select()
        model_dict[word]=model
    return model_dict

def feature_generation():
    ground_features()
    norm_features()
    polar_features()
    delta_features()
    custom_features()

def ground_features():
    asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
    asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
    asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
    asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

def norm_features():
    df_means = asl.df.groupby('speaker').mean()
    df_std = asl.df.groupby('speaker').std()

    mean = asl.df['speaker'].map(df_means['right-x'])
    std = asl.df['speaker'].map(df_std['right-x'])
    asl.df['norm-rx'] = (asl.df['right-x'] - mean) / std

    mean = asl.df['speaker'].map(df_means['right-y'])
    std = asl.df['speaker'].map(df_std['right-y'])
    asl.df['norm-ry'] = (asl.df['right-y'] - mean) / std

    mean = asl.df['speaker'].map(df_means['left-x'])
    std = asl.df['speaker'].map(df_std['left-x'])
    asl.df['norm-lx'] = (asl.df['left-x'] - mean) / std

    mean = asl.df['speaker'].map(df_means['left-y'])
    std = asl.df['speaker'].map(df_std['left-y'])
    asl.df['norm-ly'] = (asl.df['left-y'] - mean) / std

def polar_features():
    asl.df['polar-rr'] = np.sqrt(asl.df['grnd-rx'] ** 2 + asl.df['grnd-ry'] ** 2)
    asl.df['polar-rtheta'] = np.arctan2(asl.df['grnd-rx'], asl.df['grnd-ry'])
    asl.df['polar-lr'] = np.sqrt(asl.df['grnd-lx'] ** 2 + asl.df['grnd-ly'] ** 2)
    asl.df['polar-ltheta'] = np.arctan2(asl.df['grnd-lx'], asl.df['grnd-ly'])


def delta_features():
    asl.df['delta-rx'] = asl.df['right-x'].diff().fillna(0)
    asl.df['delta-ry'] = asl.df['right-y'].diff().fillna(0)
    asl.df['delta-lx'] = asl.df['left-x'].diff().fillna(0)
    asl.df['delta-ly'] = asl.df['left-y'].diff().fillna(0)

def custom_features():
    asl.df['hand_distance'] = np.sqrt(
        (asl.df['right-x'] - asl.df['left-x']) ** 2 + (asl.df['right-y'] - asl.df['left-y']) ** 2)


def train(features):
    model_selector = SelectorBIC # change as needed
    models = train_all_words(features, model_selector)
    test_set = asl.build_test(features)
    probabilities, guesses = recognize(models, test_set)
    print("Features:", features)
    print("Model Selector:", model_selector)
    show_errors(guesses, test_set)

    model_selector = SelectorDIC # change as needed
    models = train_all_words(features, model_selector)
    test_set = asl.build_test(features)
    probabilities, guesses = recognize(models, test_set)
    print("Features:", features)
    print("Model Selector:", model_selector)
    show_errors(guesses, test_set)

    model_selector = SelectorCV # change as needed
    models = train_all_words(features, model_selector)
    test_set = asl.build_test(features)
    probabilities, guesses = recognize(models, test_set)
    print("Features:", features)
    print("Model Selector:", model_selector)
    show_errors(guesses, test_set)


feature_generation()
train(features_custom)


