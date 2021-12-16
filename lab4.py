import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


data = pd.read_csv('seeds_dataset.txt', sep=r'\t+', header=None, engine='python')
data.columns = ['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry', 'length_groove', 'variety']

# (1) klasteryzacja hierarchiczna
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(data.iloc[:, :-1])

# (2) dendrogram
plt.title("Hierarchical Clustering Dendrogram")
plot_dendrogram(model, truncate_mode="level", p=3)  # plot the top p levels of the dendrogram
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# (3) policzenie średnich wartości atrybutów dla poszczególnych klastrów
clustering = AgglomerativeClustering(n_clusters=3).fit_predict(data.iloc[:, :-1])
means = data.iloc[:, :-1].groupby(clustering).mean()

# (4) wynik klasteryzacji na danych zredukowanych do dwóch wymiarów
pca = PCA(n_components=2)
data_PCA = pca.fit_transform(data.iloc[:, :-1])

# two separate plots
# fig, ax = plt.subplots()
# scatter = ax.scatter(data_PCA[:, 0], data_PCA[:, 1], c=data['variety'])
# plt.xlabel('feature #1')
# plt.ylabel('feature #2')
# plt.title('Original division by variety')
# plt.legend(handles=scatter.legend_elements()[0], labels=['Kama', 'Rosa', 'Canadian'], title='Variety of wheat')
# plt.show()
#
# fig, ax = plt.subplots()
# scatter = ax.scatter(data_PCA[:, 0], data_PCA[:, 1], c=clustering)
# plt.xlabel('feature #1')
# plt.ylabel('feature #2')
# plt.title('Hierarchical clustering')
# plt.legend(handles=scatter.legend_elements()[0], labels=['1', '2', '3'], title='Clusters')
# plt.show()

# subplots
fig, (ax1, ax2) = plt.subplots(1, 2)
scatter = ax1.scatter(data_PCA[:, 0], data_PCA[:, 1], c=data['variety'])
ax1.set_xlabel('feature #1')
ax1.set_ylabel('feature #2')
ax1.set_title('Original division by variety')
ax1.legend(handles=scatter.legend_elements()[0], labels=['Kama', 'Rosa', 'Canadian'], title='Variety of wheat')

scatter = ax2.scatter(data_PCA[:, 0], data_PCA[:, 1], c=clustering)
ax2.set_xlabel('feature #1')
ax2.set_ylabel('feature #2')
ax2.set_title('Hierarchical clustering')
ax2.legend(handles=scatter.legend_elements()[0], labels=['1', '2', '3'], title='Clusters')
plt.show()

# (5) klasteryzacja podziałowa k-means
kmeans = KMeans(n_clusters=3)
clustering_kmeans = kmeans.fit_predict(data.iloc[:, :-1])

# (6) policzenie średnich wartości atrybutów dla poszczególnych klastrów
means_kmeans = data.iloc[:, :-1].groupby(clustering_kmeans).mean() # kmeans.cluster_centers_

# (7) wynik klasteryzacji na danych zredukowanych do dwóch wymiarów
fig, (ax1, ax2) = plt.subplots(1, 2)
scatter = ax1.scatter(data_PCA[:, 0], data_PCA[:, 1], c=data['variety'])
ax1.set_xlabel('feature #1')
ax1.set_ylabel('feature #2')
ax1.set_title('Original division by variety')
ax1.legend(handles=scatter.legend_elements()[0], labels=['Kama', 'Rosa', 'Canadian'], title='Variety of wheat')

scatter = ax2.scatter(data_PCA[:, 0], data_PCA[:, 1], c=clustering_kmeans)
ax2.set_xlabel('feature #1')
ax2.set_ylabel('feature #2')
ax2.set_title('K-means clustering')
ax2.legend(handles=scatter.legend_elements()[0], labels=['1', '2', '3'], title='Clusters')
plt.show()