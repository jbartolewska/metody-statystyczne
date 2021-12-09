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

# (3) sprawdzenie wkładu poszczególnych osi
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
