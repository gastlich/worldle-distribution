from models import Country
import csv


def get_countries():
    with open("countries.csv") as csv_countries:
        reader = csv.DictReader(csv_countries)
        for row in reader:
            yield Country(
                name=row["Country"],
                latitude=row["Latitude"],
                longitude=row["Longitude"],
            )
