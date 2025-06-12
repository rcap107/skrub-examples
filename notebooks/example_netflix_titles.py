# %%
import polars as pl
import skrub

# %% Let's start by loading the table, and looking at the columns with ``TableReport``.
df = pl.read_csv("netflix_titles.csv")
skrub.TableReport(df)
# %%
# Some initial observations:
# - Title and show_id are all unique
# - Type and duration are strongly correlated, which makes sense because Movies
#   have a duration in minutes, while TV series have seasons.
# - Categorical columns include strings and free-flowing text
# - There are datetime columns
#
# We should pre-process some of the columns before starting, and then carefully
# select the categorical encoders to use for each column.

# Drop the id as it does not carry useful information
df = df.drop("show_id")

# The title should instead remain, as the TextEncoder may extract some useful
# information from it.

# %%
# Before encoding the table with |TableVectorizer|, we should first clean up the
# "duration" column.
# df = df.with_columns(
#     pl.when(pl.col("duration").str.ends_with(" min")).then(
#         pl.col("duration").str.split(" ").list.first().cast(int).alias("duration_min")
#     ),
#     pl.when(~pl.col("duration").str.ends_with(" min")).then(
#         pl.col("duration").str.split(" ").list.first().cast(int).alias("duration_seasons")
#     ),
# )

df.with_columns(
    pl.when(pl.col("duration").str.ends_with(" min"))
      .then(pl.struct(duration_min="duration"))
      .otherwise(pl.struct(duration_season="duration"))
    #   .struct.unnest()
    #   .str.split(" ").list.first().cast(int)
)

# %%
# Looking at the column contents shown in the TableReport, we can make some
# educated guesses on what encoders should be used for each.
#
# We will start by using |StringEncoder| for each categorical column, and then
# look into how TextEncoder may help.
# Additionally, we will encode datetimes with the |DatetimeEncoder|.

from skrub import StringEncoder, TextEncoder, TableVectorizer, DatetimeEncoder

categorical_encoder = StringEncoder()
datetime_encoder = DatetimeEncoder(
    resolution="day", add_weekday=True, add_total_seconds=False
)

tv = TableVectorizer(high_cardinality=categorical_encoder, datetime=datetime_encoder)

X = tv.fit_transform(df)

#%%
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

pipeline = make_pipeline(SimpleImputer(), PCA(n_components=200))

X_t = pipeline.fit_transform(X)

# %%
import numpy as np

from sklearn import metrics
from sklearn.cluster import KMeans
# %%
kmeans = KMeans(init="k-means++", n_clusters=10, n_init=4)
kmeans.fit(X_t)

# %%
import matplotlib.pyplot as plt
labels = kmeans.labels_
x_pca = PCA(n_components=2).fit_transform(X_t)
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=100)

# %%
