# %%
import polars as pl
from skrub import TableReport, Cleaner

# %%
df = pl.read_csv(
    "nyc_food_inspection/DOHMH_New_York_City_Restaurant_Inspection_Results.csv",
    ignore_errors=True,
)
df = Cleaner().fit_transform(df)
# %%
TableReport(df)
# %% We drop the following columns, identified by TableReport:
columns_to_drop = [
    "CAMIS",
    "PHONE",
    "ACTION",
    "RECORD DATE",
    "Community Board",
    "Council District",
    "Census Tract",
    "BIN",
    "BBL",
]
df = df.drop(columns_to_drop)
# %% We replace values that look like nulls in the BORO and INSPECTION DATE columns:
df = df.with_columns(
    df["INSPECTION DATE"].replace("01/01/1900", None).alias("INSPECTION DATE"),
    BORO=df["BORO"].replace("0", None),
)
# %% We filter for inspections in 2019:
df = (
    df.with_columns(pl.col("INSPECTION DATE").str.to_date("%m/%d/%Y"))
    .filter(pl.col("INSPECTION DATE").dt.year() == 2019)
    .drop_nulls(subset=["Latitude", "Longitude"])
)
# %% Now we read the airbnb dataset
df_airbnb = pl.read_csv("Airbnb_Open_Data.csv")
TableReport(df_airbnb)

# %% We drop the following columns, identified by TableReport:
df_airbnb = df_airbnb.drop(
    ["id", "host id", "country", "country code", "license"]
).drop_nulls(subset=["lat", "long", "price", "review rate number"])

# %% We can now filter for listings in 2019:
df_airbnb = df_airbnb.with_columns(
    df_airbnb["last review"].str.to_date("%m/%d/%Y")
).filter(pl.col("last review").dt.year() == 2019)

# %% Convert columns price and service fee to floats
df_airbnb = df_airbnb.with_columns(
    df_airbnb["service fee"]
    .str.replace("$", "", literal=True)
    .str.replace(",", "")
    .str.strip_chars()
    .cast(float)
    .alias("service fee"),
    price=df_airbnb["price"]
    .str.replace("$", "", literal=True)
    .str.replace(",", "")
    .str.strip_chars()
    .cast(float),
)

#%%
import seaborn as sns

sns.scatterplot(
    data=df_airbnb.to_pandas(),
    y="lat",
    x="long",
    hue="neighbourhood",
    style="neighbourhood_group"
)

# %% Now we will try to predict the price of a stay based on the food inspection data
# We will need to join the two datasets, and we will use the Joiner class for that.
# We will try to join on lat and lon to have a rough idea of the location of the listing.
# Before doing this, we will prepare the df_airbnb table into a train set for a
# ML model.

df_airbnb_sampled = df_airbnb.sample(10000)

X = df_airbnb_sampled.drop(["review rate number"])
y = df_airbnb_sampled["review rate number"]

# %% We will now build a sklearn pipeline to process the data and join the two tables
from sklearn.pipeline import make_pipeline
from skrub import Joiner, TableVectorizer, DatetimeEncoder, StringEncoder, AggJoiner
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, cross_validate

joiner = AggJoiner(
    df,
    main_key=["lat", "long"],
    aux_key=["Latitude", "Longitude"],
    cols=[
        "DBA",
    ],
    operations="count",
)
# joiner = Joiner(df, main_key=["lat", "long"], aux_key=["Latitude", "Longitude"] )

categorical = StringEncoder()
datetime = DatetimeEncoder(resolution="day")
tv = TableVectorizer(high_cardinality=categorical, datetime=datetime)

# %%
learner = HistGradientBoostingRegressor()

pipeline_joined = make_pipeline(joiner, tv, learner)
pipeline_base = make_pipeline(tv, learner)


scores = cross_val_score(pipeline_base, X, y, cv=5, scoring="r2")
print("Base pipeline scores:", scores)
print("Base pipeline mean score:", scores.mean())
scores = cross_val_score(pipeline_joined, X, y, cv=5, scoring="r2")
print("Joined pipeline scores:", scores)
print("Joined pipeline mean score:", scores.mean())
# %%

joiner = AggJoiner(
    df,
    main_key=["lat", "long"],
    aux_key=["Latitude", "Longitude"],
    cols=[
        "DBA",
    ],
    operations=[
        "count",
    ],
)

joiner.fit_transform(X)
# %%
from sklearn.cluster import DBSCAN
import numpy as np

# let's use dbscan to cluster
df = df.filter(pl.col("Latitude") != 0).filter(pl.col("Longitude") != 0)
lat, lon = df["Latitude"].to_numpy(), df["Longitude"].to_numpy()
X = np.stack([lat, lon], axis=1)
# %%
import matplotlib.pyplot as plt

# %%
from sklearn.cluster import Birch

cluster = Birch(threshold=0.00001, n_clusters=30)
labels = cluster.fit_predict(X)
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
    hue="neighbourhood",
    palette="deep",
    size=0.1,
)
# %%
sns.boxplot(data=df.to_pandas(), x="cluster", y="SCORE", size="SCORE")

# %%
df_s = df.sample(10000).drop_nulls(subset=["SCORE"])
X = df_s.drop(["SCORE"])
y = df_s["SCORE"]
# %%
pipeline_base = make_pipeline(tv, learner)

scores = cross_val_score(pipeline_base, X, y, cv=5, scoring="r2")
print("Base pipeline scores:", scores)
print("Base pipeline mean score:", scores.mean())
# %%
pipeline_base = make_pipeline(tv, learner)

scores = cross_val_score(pipeline_base, X.drop("cluster"), y, cv=5, scoring="r2")
print("Base pipeline scores:", scores)
print("Base pipeline mean score:", scores.mean())
