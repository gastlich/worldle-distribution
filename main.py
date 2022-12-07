import csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import worldle
import official

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


def get_country_tuples(countries):
    for country in countries:
        yield country.longitude, country.latitude, country.name


def plot_countries(countries):
    country_tuples = get_country_tuples(countries)

    longitudes, latitudes, labels = zip(*(country_tuples))
    plt.figure(figsize=(18, 8))
    plt.scatter(longitudes, latitudes)

    for i, label in enumerate(labels):
        plt.annotate(label, (longitudes[i], latitudes[i]))

    plt.show()

def kmeans_countries(countries):
    country_tuples = get_country_tuples(countries)

    longitudes, latitudes, labels = zip(*(country_tuples))

    kmeans = KMeans(n_clusters=len(TEAMMEMBERS))
    kmeans.fit(list(zip(longitudes, latitudes)))
    plt.figure(figsize=(18, 8))
    plt.scatter(longitudes, latitudes, c=kmeans.labels_)

    for i, label in enumerate(labels):
        plt.annotate(label, (longitudes[i], latitudes[i]))

    plt.show()


countries = worldle.get_countries()
# plot_countries(countries)
kmeans_countries(countries)
