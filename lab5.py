import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix

attr = ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'class']
dataset = pd.read_csv('yeast.data', names=attr, delim_whitespace=True)
dataset['class'] = LabelEncoder().fit_transform(dataset['class'])

X = dataset.iloc[:, 0:7].values
y = dataset.iloc[:, 8].values

# (1) i (2)
# losowy podział zbioru
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
#                                                     random_state=0)

# klasyfikacja z walidacją krzyżową
print('CLassification using kNN')
# wersja nr 1
print('VERSION #1')
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
print('VERSION #2')
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

# (3) klasyfikacja przy pomocy drzewa
print('\nCLassification with decision tree')
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
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 8), dpi=300)
    tree.plot_tree(classifier, max_depth=3, feature_names=dataset.columns[:-1].values, filled=True, fontsize=5)
    plt.savefig('lab5_tree_k{}.png'.format(k))
    # plt.show()

# (4) inne klasyfikatory sklearn z domyślnymi ustawieniami
# RandomForestClassifier
print('\nCLassification with random forest')
for k in range(3, 7):
    print(' k = {}'.format(k))
    kf = KFold(n_splits=k, random_state=None)
    classifier = RandomForestClassifier()
    acc = cross_val_score(classifier, X, y, cv=kf)
    print('     accuracy of each fold - {}'.format(acc))
    print('     avg accuracy : {}'.format(acc.mean()))

# Support Vector Machines (SVM)
print('\nCLassification with SVM')
for k in range(3, 7):
    print(' k = {}'.format(k))
    kf = KFold(n_splits=k, random_state=None)
    classifier = SVC()
    acc = cross_val_score(classifier, X, y, cv=kf)
    print('     accuracy of each fold - {}'.format(acc))
    print('     avg accuracy : {}'.format(acc.mean()))

# Naive Bayes
print('\nCLassification with Gaussian/Multimonial Naive Bayes')
for k in range(3, 7):
    print(' k = {}'.format(k))
    kf = KFold(n_splits=k, random_state=None)
    classifier = MultinomialNB()
    acc = cross_val_score(classifier, X, y, cv=kf)
    print('     accuracy of each fold - {}'.format(acc))
    print('     avg accuracy : {}'.format(acc.mean()))

# Linear Discriminant Analysis (LDA)
print('\nCLassification with LDA')
for k in range(3, 7):
    print(' k = {}'.format(k))
    kf = KFold(n_splits=k, random_state=None)
    classifier = LinearDiscriminantAnalysis()
    acc = cross_val_score(classifier, X, y, cv=kf)
    print('     accuracy of each fold - {}'.format(acc))
    print('     avg accuracy : {}'.format(acc.mean()))

# evaluation
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))