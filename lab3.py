import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

data = pd.read_csv('seeds_dataset.txt', sep=r'\t+', header=None, engine='python')
data.columns = ['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry', 'length_groove', 'variety']

# (1) redukcja wymiarowości zbioru do N=2 z wykorzystaniem PCA
pca = PCA(n_components=2)
data_PCA = pca.fit_transform(data.iloc[:, :-1])

# (2) wykreślenie danych w przestrzeni dwuwymiarowej
fig, ax = plt.subplots()
scatter = ax.scatter(data_PCA[:, 0], data_PCA[:, 1], c=data['variety'])
plt.xlabel('feature #1')
plt.ylabel('feature #2')
plt.title('PCA')
plt.legend(handles=scatter.legend_elements()[0], labels=['Kama', 'Rosa', 'Canadian'], title='Variety of wheat')
plt.show()

# (3) sprawdzenie wkładu poszczególnych osi (nowych!)
# PCA ustala liczbę komponentów poprzez min(liczba wierszy, liczba kolumn)
# i explained_variance_ratio_ tłumaczy jaki te nowe komponenty stworzone przez PCA mają wkład
# nie można tego bezpośrednio odnieść do cech zadanych na początku
# (to mówi nam, że PC1 explains ..%, PC2 ...)
# natomiast w components_ można zobaczyć dalej jaki wkład w poszczególne te PCX mają początkowe cechy
# https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis
pca = PCA().fit(data.iloc[:, :-1])
print(pca.explained_variance_ratio_)

# (4) wykorzystanie t-SNE do redukcji wymiarowości
data_TSNE = TSNE(n_components=2).fit_transform(data.iloc[:, :-1])
fig, ax = plt.subplots()
scatter = ax.scatter(data_TSNE[:, 0], data_TSNE[:, 1], c=data['variety'])
plt.xlabel('feature #1')
plt.ylabel('feature #2')
plt.title('t-SNE')
plt.legend(handles=scatter.legend_elements()[0], labels=['Kama', 'Rosa', 'Canadian'], title='Variety of wheat')
plt.show()
