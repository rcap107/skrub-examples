# %%
import polars as pl
import skrub
from skrub import TableReport

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

# %%
data = skrub.var("data", df_airbnb).sample(5000)
X = data.drop("review rate number").skb.mark_as_X()
y = data["review rate number"].skb.mark_as_y()
# %%

from skrub import (
    MinHashEncoder,
    StringEncoder,
)
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
        "minhash": MinHashEncoder(n_components=30),
        "string": StringEncoder(n_components=30),
    },
    name="encoder_categorical",
)

# %%
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

pf = make_pipeline(SimpleImputer(), PolynomialFeatures(2, interaction_only=True))
# %%
text_like_columns = ["NAME", "host name", "neighbourhood", "house_rules"]
categorical_like_columns = [
    "host_identity_verified",
    "neighbourhood group",
    "cancellation_policy",
    "room type",
]

text_like = X.select(text_like_columns).skb.apply(encoder_text_like)
categorical_like = X.select(categorical_like_columns).skb.apply(encoder_categorical)
rest = X.drop(categorical_like_columns + text_like_columns).skb.apply(
    skrub.TableVectorizer(numeric=pf)
)
transformed = text_like.skb.concat_horizontal([text_like, categorical_like, rest])

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

hgb = HistGradientBoostingRegressor(
    learning_rate=skrub.choose_float(0.01, 0.09, log=True, name="lr")
)
predictions = transformed.skb.apply(
    hgb,
    y=y,
)

# %%

search = predictions.skb.get_randomized_search(
    n_jobs=4, fitted=True, n_iter=16, random_state=1, scoring="r2"
)
# %%
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import Birch

kbins = KBinsDiscretizer(
    encode="onehot-dense",
    n_bins=skrub.choose_int(10, 30, n_steps=5, name="kbins_nbins"),
)

birch = Birch(threshold=0.002, n_clusters=30)

# %%
new_transformed = X.skb.select(["lat", "long"]).skb.apply(birch)
# %%
cluster = new_transformed.skb.applied_estimator()
# %%
df_restaurants = (
    pl.read_csv(
        "nyc_food_inspection/DOHMH_New_York_City_Restaurant_Inspection_Results.csv",
        ignore_errors=True,
    )
    .drop_nulls(subset=["SCORE"])
    .sample(5_000)
)
# %%
restaurants = skrub.var("restaurants", df_restaurants)
restaurants_coords =restaurants.skb.select(["Latitude", "Longitude"]).skb.apply(SimpleImputer())
# %%
re=restaurants_coords.skb.apply(cluster)
# %%
