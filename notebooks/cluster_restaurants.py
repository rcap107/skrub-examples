# %%
import polars as pl
from skrub import TableReport
import numpy as np

# %%
df = pl.read_csv(
    "nyc_food_inspection/DOHMH_New_York_City_Restaurant_Inspection_Results.csv",
    ignore_errors=True,
)
# %%
average_latitude = df["Latitude"].mean()
average_longitude = df["Longitude"].mean()

# %%
df = df.filter(pl.col("Latitude") != 0).filter(pl.col("Longitude") != 0)
sample_1 = df.sample(10000).select(pl.col("Latitude"), pl.col("Longitude"))
sample_2 = df.sample(10000).select(pl.col("Latitude"), pl.col("Longitude"))


def distance(lat1, lon1, lat2, lon2, average_latitude):
    return (((lat1 - lat2) * np.cos(average_latitude)) ** 2 + (lon1 - lon2) ** 2) ** 0.5


# %%
d = distance(
    sample_1["Latitude"],
    sample_1["Longitude"],
    sample_2["Latitude"],
    sample_2["Longitude"],
    average_latitude,
)
# %%
dbscan = DBSCAN(eps=0.01, min_samples=10)
labels = dbscan.fit_predict(df.to_pandas()[["Latitude", "Longitude"]])

# plt.scatter(lon, lat, c=labels)
# %%
l = pl.Series(
    name="cluster",
    values=labels,
)
df = df.with_columns(cluster=l)
# %%
import seaborn as sns

sns.scatterplot(
    data=df.to_pandas(),
    x="Longitude",
    y="Latitude",
    hue="cluster",
    palette="deep",
    size=0.1,
)
# %%
