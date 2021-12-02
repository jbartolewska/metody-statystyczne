import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.dates as mdates

df = pd.read_csv('owid-covid-data.csv')
df = df.loc[df['location'] == 'Poland']

# (1a) liczba przypadków brakujących
df_worthness = df.isna().sum()

# (1b) podstawowe statystyki dla wybranych kolumn
df = df[['date', 'new_cases', 'new_deaths', 'hosp_patients', 'new_tests']]
df.describe()

# (2) uzupełnianie brakujących elementów strategią k-NN
imputer = KNNImputer(n_neighbors=2)
df_rep_nans = df.copy()
df_rep_nans.iloc[:, 1:] = imputer.fit_transform(df.iloc[:, 1:])

df_rep_nans['date'] = pd.to_datetime(df['date'])
# (3) jednowymiarowe wykrywanie elementów odstających
for (column_name, column_data) in df_rep_nans.iloc[:, 1:].iteritems():
    plt.plot(df_rep_nans['date'], column_data, color='k', zorder=1)
    plt.scatter(df_rep_nans['date'][column_data[np.abs(stats.zscore(column_data)) > 3].index],
                column_data[np.abs(stats.zscore(column_data)) > 3], facecolors='none', edgecolors='r', zorder=2)
    plt.xlabel('day')
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.ylabel('#' + column_name)
    plt.show()

# (4) wielowymiarowe wykrywanie elementów odstających -> Local Outlier Factor
clf = LocalOutlierFactor(n_neighbors=10)
mask = clf.fit_predict(df_rep_nans.iloc[:, 1:])
for (column_name, column_data) in df_rep_nans.iloc[:, 1:].iteritems():
    plt.plot(df_rep_nans['date'], column_data, color='k', zorder=1)
    plt.scatter(df_rep_nans['date'][column_data[mask == -1].index],
                column_data[mask == -1], facecolors='none', edgecolors='r', zorder=2)
    plt.xlabel('day')
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.ylabel('#' + column_name)
    plt.show()
