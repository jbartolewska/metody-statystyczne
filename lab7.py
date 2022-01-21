from surprise import Dataset
import pandas as pd
from surprise.reader import Reader
from surprise.prediction_algorithms.knns import KNNWithMeans
from collections import defaultdict
from surprise.model_selection import train_test_split, KFold, cross_validate
from surprise import get_dataset_dir
import io
import numpy as np
from numpy.random import default_rng


def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def read_item_names():
    """Read the u.item file from MovieLens 100-k dataset and return two
    mappings to convert raw ids into movie names and movie names into raw ids.
    """

    file_name = get_dataset_dir() + '/ml-100k/ml-100k/u.item'
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid

# Pobranie zbioru danych MovieLens-100k
# format: user item rating timestamp
data = Dataset.load_builtin('ml-100k')

# # Definicja własnego (~randomowego) wektora ocen filmów (50 el.)
# # - oceny brane z rozkładu jednostajnego (gaussowski byłby bardziej odpowiedni)
# rng = default_rng(123)
# ratings_dict = {'itemID': np.repeat(np.arange(1, 6), 10),
#                 'userID': np.tile(np.arange(1, 11), 5),
#                 'rating': np.concatenate([rng.integers(low=1, high=5, size=10, endpoint=True) for x in range(5)]).ravel()}
# df = pd.DataFrame(ratings_dict)
# df = df.sample(frac=1)
#
# reader = Reader(rating_scale=(1, 5))
# data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

# System rekomendacyjny oparty o metodę ważonych k-najbliższych sąsiadów
trainset, testset = train_test_split(data)
algo = KNNWithMeans()
predictions = algo.fit(trainset).test(testset)

# # Wersja alternatywna z walidacją krzyżową
# # (1)
# algo = KNNWithMeans()
# kf = KFold(n_splits=5)
# for trainset, testset in kf.split(data):
#     predictions = algo.fit(trainset).test(testset)
#
# # (2)
# algo = KNNWithMeans()
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
#
# # Wersja alternatywna z treningiem na podstawie pełnego zbiory i
# # testowaniem z wykorzystaniem antyzbioru
# # (danych, które nie pojawiły się podczas uczenia)
# algo = KNNWithMeans()
# trainset = data.build_full_trainset()
# algo.fit(trainset)
#
# testset = trainset.build_anti_testset()
# predictions = algo.test(testset)

# 10 najwyższych rekomendacji dla każdego użytkownika
top_n = get_top_n(predictions, n=10)

rid_to_name, name_to_rid = read_item_names()
for uid, user_ratings in top_n.items():
    # print(uid, [iid for (iid, _) in user_ratings])
    print(uid, [rid_to_name[iid] for (iid, _) in user_ratings])

