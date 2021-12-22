import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import tree

attr = ['Sequence Name', 'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc']
dataset = pd.read_csv('yeast.data', names=attr, delim_whitespace=True)
dataset['nuc'] = LabelEncoder().fit_transform(dataset['nuc'])

X = dataset.iloc[:, 0:7].values
y = dataset.iloc[:, 8].values

# (1)
# losowy podział zbioru
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
#                                                     random_state=0)

# klasyfikacja z walidacją krzyżową
# wersja nr 1
print('WERSJA #1')
metrics = ['minkowski', 'euclidean', 'cosine']
for metric in metrics:
    print('Metric - {}'.format(metric))
    for k in range(3, 7):
        print(' k = {}'.format(k))
        kf = KFold(n_splits=k, random_state=None)
        classifier = KNeighborsClassifier(n_neighbors=5, metric=metric)
        acc_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            acc = accuracy_score(y_pred, y_test)
            acc_scores.append(acc)

        print('     accuracy of each fold - {}'.format(acc_scores))
        print('     avg accuracy : {}'.format(sum(acc_scores) / k))

# wersja nr 2
print('WERSJA #2')
metrics = ['minkowski', 'euclidean', 'cosine']
for metric in metrics:
    print('Metric - {}'.format(metric))
    for k in range(3, 7):
        print(' k = {}'.format(k))
        kf = KFold(n_splits=k, random_state=None)
        classifier = KNeighborsClassifier(n_neighbors=5, metric=metric)
        acc = cross_val_score(classifier, X, y, cv=kf)

        print('     accuracy of each fold - {}'.format(acc))
        print('     avg accuracy : {}'.format(acc.mean()))

# jak zwiększyć accuracy?
# więcej sąsiadów lub podział zbioru na większą liczbę podzbiorów

# (2) klasyfikacja przy pomocy drzewa
print('CLassification with decision tree')
for k in range(3, 7):
    print(' k = {}'.format(k))
    kf = KFold(n_splits=k, random_state=None)
    classifier = tree.DecisionTreeClassifier()
    # acc = cross_val_score(classifier, X, y, cv=kf)
    # print('     accuracy of each fold - {}'.format(acc))
    # print('     avg accuracy : {}'.format(acc.mean()))

    acc_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        acc = accuracy_score(y_pred, y_test)
        acc_scores.append(acc)

    print('     accuracy of each fold - {}'.format(acc_scores))
    print('     avg accuracy : {}'.format(sum(acc_scores) / k))
    tree.plot_tree(classifier)

    