# %% [raw]
# ---
# title: Applying Skrub expressions to the NYC food inspection dataset
# author: Riccardo Cappuzzo
# execute:
#   echo: true
#   output: true
# ---
"""In this notebook we will explore the NYC food inspection dataset and use all the
tools available in the skrub library to prepare the data for a ML model.
We will make use of the TableReport to look at the data and identify columns that
may need particular attention, and then use the skrub expressions to build a complex
data preparation and training pipline.

The dataset that we are considering includes the inspection results for restaurants
in the different neighborhoods of New York City, over multiple years.
We will use the inspection result (``SCORE``) as prediction target.
"""

# %%
import polars as pl
from skrub import TableReport, Cleaner
import skrub

# %%
# We start by reading the NYC food inspection dataset.
df = (
    pl.read_csv(
        "nyc_food_inspection/DOHMH_New_York_City_Restaurant_Inspection_Results.csv",
        ignore_errors=True,
    )
    .drop_nulls(subset=["SCORE"])
    .sample(5_000)  # sampling to avoid issues with scalability
)
# %% [markdown]
# We then use the TableReport to look at the data distribution and decide what
# to do next.
# %%
TableReport(df)

# %% [markdown]
# % Thanks to the TableReport we can already make some decisions about the data.
# The data includes various columns that are not useful for our analysis.
# The ZIPCODE and PHONE columns have been converted to integer columns, but they
# should be treated as categorical variables. PHONE should be dropped,
# but ZIPCODE may be useful if treated as a categorical variable.
# RECORD DATE is a constant, so we can drop it.
# CAMIS, Community Board, Council District, Census Tract, BIN, and BBL all appear
# to be unique identifiers, so we can drop them as well.
# By looking at the outliers in the TableReport we can see that Latitude and Longitude
# have outliers that are likely to be missing values.
# Similarly, the INSPECTION DATE column contains a lot of dates ("01/01/1900") that are
# likely to be missing values.
#
# %%
# We begin by defining the X and y variables.
full_data = skrub.var("data", df)
X = full_data.drop("SCORE").skb.mark_as_X()
y = full_data["SCORE"].skb.mark_as_y()
# %% We drop the following columns, identified by TableReport:
columns_to_drop = [
    "CAMIS",
    "PHONE",
    "RECORD DATE",
    "Community Board",
    "Council District",
    "Census Tract",
    "BIN",
    "BBL",
]
X = X.drop(columns_to_drop)

# %% [markdown]
# We replace values that look like nulls in the BORO and INSPECTION DATE columns:
# Note: by using skrub.deferred we can define the function and have it execute lazily.
# This will also collapse the computation graph for simplicity. If you comment out
# the skrub.deferred decorator, you will be able to see the full computation graph.


# %%
@skrub.deferred
def replace_nulls(df):
    return df.with_columns(
        df["INSPECTION DATE"].replace("01/01/1900", None).alias("INSPECTION DATE"),
        BORO=df["BORO"].replace("0", None),
    )


@skrub.deferred
def convert_to_cat(df):
    return df.with_columns(ZIPCODE=pl.col("ZIPCODE").cast(str))


@skrub.deferred
def replace_null_coords(df):
    return df.fill_null(0).with_columns(
        Latitude=pl.when((df["Latitude"] == 0) & (df["Longitude"] == 0))
        .then(pl.col("Latitude").mean())
        .otherwise(pl.col("Latitude")),
        Longitude=pl.when((df["Latitude"] == 0) & (df["Longitude"] == 0))
        .then(pl.col("Longitude").mean())
        .otherwise(pl.col("Longitude")),
    )


# %% [markdown]
# Deferred versions can be called like regular functions by taking an expression
# and possibly more arguments, or they can be called using `.skb.apply_func` if
# their only argument is the expression itself (like in this case).
# %%
X = replace_nulls(X)
# X = X.skb.apply_func(replace_nulls)
X = convert_to_cat(X)
# X = X.skb.apply_func(convert_to_cat)
X = X.skb.apply_func(replace_null_coords)

# %% [markdown]
# We now define various encoders to choose from to encode the categorical variables.
# "Text-like" columns include values that are more "natural language-looking",
# such as names or descriptions.
# "Categorical-like" columns contain single strings or categories.
#
# We use `skrub.choose_from` to easily create parameter choices that can be
# evaluated and tested using grid or randomized search.

# %%
from skrub import (
    MinHashEncoder,
    StringEncoder,
)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

encoder_text_like = skrub.choose_from(
    {
        "string": StringEncoder(n_components=30),
        "minhash": MinHashEncoder(n_components=30),
    },
    name="encoder_text_like",
)

encoder_categorical = skrub.choose_from(
    {
        "ordinal": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        "minhash": MinHashEncoder(n_components=30),
    },
    name="encoder_categorical",
)

# %% [markdown]
# In the following block we apply the different encoders to each subset of columns
# and a "leftover" transformer that runs on everything but the columns that have
# already been handled.

# %%
cols_text_like = ["VIOLATION DESCRIPTION", "DBA", "CUISINE DESCRIPTION"]
cols_categorical = ["NTA", "ZIPCODE"]
columns_already_handled = cols_text_like + cols_categorical
transformed = (
    X.skb.apply(
        encoder_text_like,
        cols=cols_text_like,
    )
    .skb.apply(encoder_categorical, cols=cols_categorical)
    .skb.apply(
        skrub.TableVectorizer(),
        exclude_cols=columns_already_handled,
    )
    .skb.apply(SimpleImputer(), cols=["Latitude", "Longitude"])
)

# %% [markdown]
# Now we can define the predictor, and use a HGB regressor with a choice of parameters
# for the learning rate.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

hgb = HistGradientBoostingRegressor(
    learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="lr")
)
predictions = transformed.skb.apply(
    hgb,
    y=y,
)
# %% [markdown]
# After building the entire pipeline, we can run a randomized search with the given
# set of hyperparameters to test, and look at the result.

# %%
search = predictions.skb.get_randomized_search(
    n_jobs=4, fitted=True, n_iter=16, random_state=1, scoring="r2"
)
# %% [markdown]
# By setting the parameter  ``fitted`` to True the random search will run
# cross-validation on each configuration, and then report the average score of
# each configuration across the crossvalidation folds.
# %%
# We can access the results directly using ``.results``.
search.results_
# %%
# We can also plot a critical difference plot to identify the most important parameters
search.plot_results()
# %% [markdown]
# For the sake of comparison, let's compare the performance of our pipeline to that
# obtained uisng the `skrub.tabular_learner`.
# %%
from skrub import tabular_learner
from sklearn.model_selection import cross_validate

tl = tabular_learner(HistGradientBoostingRegressor())
X_ = df.drop("SCORE")
y_ = df["SCORE"]
results_tl = cross_validate(tl, X_, y_)

# %%
print(f"Expressions: {search.best_score_:.4f}")
print(f'Tabular learner: {results_tl["test_score"].mean():.4f}')

# %% [markdown]
# In the next section of the notebook we will perform feature engineering by
# adding cluster-based features to the dataset. For this, we will use Birch as
# clustering method.

# %%
from sklearn.cluster import Birch
from sklearn.preprocessing import KBinsDiscretizer

# %%
coords = X.skb.select(["Latitude", "Longitude"])


# %%
@skrub.deferred
def get_clusters(df, cluster):
    return cluster.fit_predict(df)


clustering_method = skrub.choose_from(
    {
        "birch": Birch(
            threshold=0.002,
            n_clusters=skrub.choose_int(10, 100, n_steps=5, name="birch_clusters"),
        )
    },
    name="clustering_method",
)

# %%
clustered_coordinates = coords.skb.apply(clustering_method)
new_transformed = transformed.skb.concat_horizontal([clustered_coordinates])
# %%

hgb = HistGradientBoostingRegressor(
    learning_rate=skrub.choose_float(0.01, 0.09, log=True, n_steps=3, name="lr")
)
new_predictions = new_transformed.skb.apply(
    hgb,
    y=y,
)
new_search = new_predictions.skb.get_randomized_search(
    n_jobs=4, fitted=True, n_iter=16, random_state=1, scoring="r2"
)
# %%
print(f"Expressions: {search.best_score_:.4f}")
print(f"Expressions with cluster features: {new_search.best_score_:.4f}")