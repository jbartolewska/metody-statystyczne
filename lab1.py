import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# (1) wczytanie danych dla Polski
df = pd.read_csv('owid-covid-data.csv')
df = df.loc[df['location'] == 'Poland']

# (2) weryfikacja zmiennych pod względem liczby elementów brakujących
# df['total_cases'].isna().sum()
df_worthness = df.isna().sum()

# df.dropna(axis=1, inplace=True) # usunięcie kolumn z wartościami NaN
# df.dropna(axis=0, inplace=True) # usunięcie wierszy z wartościami NaN

# (3) podsumowania statystyczne dla wybranych kolumn
df = df[['date', 'new_cases', 'new_deaths', 'hosp_patients', 'new_tests']]
df.describe()

# (4) agregacja danych do tygodni
# df['date'] = pd.to_datetime(df['date'])
# df['week'] = df['date'].dt.week # the week ordinal of the year
# df['year'] = df['date'].dt.year

df['date'] = pd.to_datetime(df['date']) - pd.to_timedelta(7,
                                                          unit='d')  # subtracting week - specific behaviour of pd.Grouper with W-MON
df_weekly = df.groupby([pd.Grouper(key='date', freq='W-MON')]).sum().reset_index().sort_values(
    'date')  # with weekly frequency (Mondays)

# (5) wykres przebiegu w funkcji czasu dla każdej kolumny
for (column_name, column_data) in df_weekly.iloc[:, 1:].iteritems():
    plt.plot(df_weekly['date'], column_data)
    plt.xlabel('week')
    plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.ylabel('#' + column_name)
    plt.show()

# (6) normalizacja zbioru poprzez strategię min-max
df_weekly_norm = pd.DataFrame(MinMaxScaler().fit_transform(df_weekly.iloc[:, 1:]), index=df_weekly.iloc[:, 1:].index,
                              columns=df_weekly.iloc[:, 1:].columns)
df_weekly_norm.insert(loc=0, column='date', value=df_weekly['date'])

# (7) dane w formie tydzień do tygodnia
colors = df_weekly['date'][1:].index
for column in df_weekly.iloc[:, 1:]:
    plt.scatter(df_weekly[column][1:], df_weekly[column][:-1], c=colors, cmap='YlOrBr')

    cbar = plt.colorbar()
    cbar.ax.get_yaxis().set_ticks([])
    cbar.ax.text(0, 91.5, 'newest')
    cbar.ax.text(0, -2.5, 'oldest')

    plt.xlabel('current week')
    plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.ylabel('last week')
    plt.title('#' + column)
    plt.show()
