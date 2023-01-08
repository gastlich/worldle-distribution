import csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import worldle
import official
from collections import Counter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np

TEAMMEMBERS = {
    "Nihar",
    "Prat",
    "Anitha",
    "Suresh",
    "Sonam",
    "Sandeep",
    "Jacek",
    "Thomas",
}

N_CLUSTERS = len(TEAMMEMBERS)


def get_country_tuples(countries):
    for country in countries:
        yield country.longitude, country.latitude, country.name


def plot_countries(longitudes, latitudes, colours=None):
    plt.figure(figsize=(18, 8))
    plt.scatter(longitudes, latitudes, c=colours)

    for i, label in enumerate(labels):
        plt.annotate(label, (longitudes[i], latitudes[i]))

    plt.show()


def make_clusters_even(clusters, cordinates):
    cluster_size = int(np.ceil(len(cordinates) / N_CLUSTERS))

    centers = clusters.cluster_centers_
    centers = (
        centers.reshape(-1, 1, cordinates.shape[-1])
        .repeat(cluster_size, 1)
        .reshape(-1, cordinates.shape[-1])
    )
    distance_matrix = cdist(cordinates, centers)
    clusters = linear_sum_assignment(distance_matrix)[1] // cluster_size

    return clusters


def even_kmeans_countries(longitudes, latitudes):
    cordinates = np.array(list(zip(longitudes, latitudes)))

    clusters = KMeans(n_clusters=N_CLUSTERS)
    clusters.fit(cordinates)

    return make_clusters_even(clusters, cordinates)


countries = worldle.get_countries()
country_tuples = get_country_tuples(countries)
longitudes, latitudes, labels = zip(*(country_tuples))


clusters = even_kmeans_countries(longitudes, latitudes)
plot_countries(longitudes, latitudes, colours=clusters)
