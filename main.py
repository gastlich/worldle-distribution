import csv
from itertools import groupby
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import worldle
import official
from collections import Counter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np

TEAMMEMBERS = [
    "Nihar",
    "Prathap",
    "Anitha",
    "Suresh",
    "Sonam",
    "Sandeep",
    "Jacek",
    "Blake",
]

N_CLUSTERS = len(TEAMMEMBERS)


def get_country_tuples(countries):
    for country in countries:
        yield country.longitude, country.latitude, country.name


def plot_countries(longitudes, latitudes, labels, colours=None, title=None):
    plt.figure(figsize=(18, 8))
    plt.scatter(longitudes, latitudes, c=colours)

    if title:
        plt.title(title)

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

    clusters = KMeans(n_clusters=N_CLUSTERS, random_state=1)
    clusters.fit(cordinates)

    return make_clusters_even(clusters, cordinates)


def group_by_cluster(longitudes, latitudes, labels, clusters):
    key_function = lambda x: x[-1]

    yield from groupby(
        sorted(zip(longitudes, latitudes, labels, clusters), key=key_function),
        key=key_function,
    )


countries = worldle.get_countries()
country_tuples = get_country_tuples(countries)
longitudes, latitudes, labels = zip(*(country_tuples))

clusters = even_kmeans_countries(longitudes, latitudes)
plot_countries(
    longitudes,
    latitudes,
    labels,
    colours=clusters,
    title=f"Even cluster distribution among {len(TEAMMEMBERS)} team members.",
)

grouped_countries = group_by_cluster(longitudes, latitudes, labels, clusters)

for i, countries in grouped_countries:
    countries = list(countries)
    unzipped_countries = zip(*list(countries))
    longitudes, latitudes, labels, cluster = unzipped_countries

    member = TEAMMEMBERS[i]
    plot_countries(longitudes, latitudes, labels, title=member)

    print(f"➡️  {member} is going to learn the following {len(countries)} countries:")
    for country in countries:
        print(f"{country[2]} ({country[0]}, {country[1]})")

    print()
