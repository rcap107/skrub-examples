# %%
import polars as pl
import skrub

from skrub import TableReport, Cleaner
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# %%
df = pl.read_parquet("../_scratch/traffic_violations_train.parquet")
df = df.filter(~pl.col("violation_type").is_in(["SERO", "ESERO"]))
data = skrub.var("data", df)#.skb.subsample(1000)
# We first convert X and y into skrub expressions.
X = data.drop("violation_type").skb.mark_as_X()
y = data["violation_type"].skb.mark_as_y()

y = y.to_numpy().reshape(-1, 1).skb.apply(OrdinalEncoder())


# %%
# We can convert columns that contain dates and drop uninformative columns using the Cleaner.
c = Cleaner(drop_null_fraction=0.9, drop_if_constant=True, numeric_dtype="float32")

X_ = X.skb.apply(c)

X_ = X_.sort("date", descending=False)

# %%
import skrub.selectors as s

# Create a column selector with arbitrary logic.
binary = s.filter(lambda col: col.drop_nulls().n_unique() == 2)


# %%
def convert_to_binary(df):
    for col in df.columns:
        df = df.with_columns(
            pl.col(col).cast(pl.Categorical).cast(pl.Boolean, strict=False).alias(col)
        )
    return df

binary_cols = X_.skb.select(binary).skb.apply_func(convert_to_binary)

converted = X_.skb.select(~binary).skb.concat([binary_cols], axis=1)
# converted
# %%
# Drop more columnns that are not informative.
preprocessed = converted.drop(
    "seqid",
    "geolocation",
)

# %%
# We can now use the skrub encoders to add new features to the table.
from skrub import TableVectorizer, DatetimeEncoder, StringEncoder
from sklearn.preprocessing import TargetEncoder, StandardScaler, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

cat_encoder = skrub.choose_from(
    {
        # "target": TargetEncoder(),
        "string": StringEncoder(n_components=5),
    },
    name="cat_encoder",
)
datetime_encoder = DatetimeEncoder(
    resolution="day",
    add_weekday=True,
    add_total_seconds=False,
    periodic_encoding=skrub.choose_from(
        [None, "spline", "circular"], name="periodic_encoding"
    ),
)

num_encoder = skrub.choose_from(
    {
        "kbins": make_pipeline(
            SimpleImputer(),
            KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile"),
        ),
        "passthrough": SimpleImputer(),
    },
    name="num_encoder",
)

postprocess = make_pipeline(
    SimpleImputer(),
)

high_cardinality = s.string() - s.cardinality_below(40)

leftover = s.all() - s.any_date() - high_cardinality - s.numeric()

dates = preprocessed.skb.select(cols=s.any_date()).skb.apply(datetime_encoder)
strings = preprocessed.skb.select(cols=s.string() - s.cardinality_below(40)).skb.apply(
    cat_encoder,
)
numbers = preprocessed.skb.select(cols=s.numeric()).skb.apply(
    num_encoder, 
)

everything_else = preprocessed.skb.select(
    cols=leftover
).skb.apply(TableVectorizer(), cols=leftover).skb.apply(postprocess)


encoded = (
    everything_else.skb.concat([dates, strings, numbers], axis=1)
)
encoded
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

cv = TimeSeriesSplit(
    n_splits=5,
    max_train_size=None,
    gap=0,
)
#%%
model = skrub.choose_from(
    {
        # "logistic": LogisticRegression(max_iter=500,
        #     # max_iter=skrub.choose_int(100, 1000, log=True, name="max_iter"),
        #     C=skrub.choose_float(0.5, 10, log=True, name="C", n_steps=3),
        # ),
        "hgb": HistGradientBoostingClassifier(
            learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="lr")
        ),
    },
    name="model",
)
# %%
predicted = encoded.skb.apply(model, y=y.ravel())

# %%
search = predicted.skb.get_randomized_search(
    n_jobs=4, fitted=True, n_iter=8, random_state=1, scoring="roc_auc", cv=cv,
)
# %%
search.plot_results()
# %%
search.detailed_results_   
# %%
import pickle

saved_model = pickle.dumps(search.best_pipeline_)

# %%
new_df = pl.read_parquet("../_scratch/traffic_violations_test.parquet")
new_df = new_df.filter(~pl.col("violation_type").is_in(["SERO", "ESERO"]))
new_data = skrub.var("data", new_df)#.skb.subsample(1000)

loaded_model = pickle.loads(saved_model)
pred = loaded_model.score({"data": new_df})
# %%
skrub.cross_validate(search, predicted.skb.get_data(), cv=cv)
# %%
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier()

dummy.fit(df, df["violation_type"])
# %%
dummy.score(new_df, new_df["violation_type"])
# %%
skrub.cross_validate(loaded_model, {"data": new_df})
# %%
