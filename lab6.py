from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

(X_train, y_train) = fetch_20newsgroups(subset="train", random_state=42, return_X_y=True)
(X_test, y_test) = fetch_20newsgroups(subset="test", random_state=42, return_X_y=True)

# hashing vectorizer
vectorizer = HashingVectorizer(stop_words="english", alternate_sign=False)
print('Hasing Vectorizer')

# # tfid vectorizer
# vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words="english")
# print('TF-IDF Vectorizer')

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

metrics = ['minkowski', 'euclidean', 'cosine']
for metric in metrics:
    print('Metric - {}'.format(metric))
    for k in [1, 5, 10]:
        print(' neighbours = {}'.format(k))
        classifier = KNeighborsClassifier(n_neighbors=k, metric=metric)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        acc = accuracy_score(y_pred, y_test)

        print('     accuracy : {}'.format(acc))