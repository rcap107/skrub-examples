# %%
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

from skrub import TableVectorizer, StringEncoder, TextEncoder

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_validate
from sklearn.linear_model import RidgeCV

import numpy as np
from skrub import TableReport
import skrub
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from xgboost import XGBRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# %%
df = pl.read_csv("laptop_data.csv")
# %%
TableReport(df)
# %%
# Some preprocessing is needed:
# - Some of the columns should be treated as floats, but they include strings
# - There is an index column that should be dropped

df = df.with_columns(
    Ram=pl.col("Ram").str.strip_suffix("GB").cast(int),
    Weight=pl.col("Weight").str.strip_suffix("kg").cast(float),
).drop("")
# %%
X = df.drop("Price").to_pandas()
y = df["Price"].to_pandas()
# %% Without expressions:
tv_base = TableVectorizer(
    low_cardinality=OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    high_cardinality=OneHotEncoder(handle_unknown="ignore", sparse_output=False),
)
tv_text = TableVectorizer(
    high_cardinality=TextEncoder(),
)
tv_string = TableVectorizer(
    high_cardinality=StringEncoder(),
)

model_gbdt = make_pipeline(tv_base, HistGradientBoostingRegressor())
model_text = make_pipeline(tv_text, HistGradientBoostingRegressor())
model_string = make_pipeline(tv_string, HistGradientBoostingRegressor())


# model_gbdt = make_pipeline(tv_base, StandardScaler(), SimpleImputer(),  RidgeCV())
# model_text = make_pipeline(tv_text, StandardScaler(), SimpleImputer(),  RidgeCV())
# model_string = make_pipeline(tv_string, StandardScaler(), SimpleImputer(),  RidgeCV())


# %%
result_base = cross_validate(model_gbdt, scoring="r2", X=X, y=y)
result_text = cross_validate(model_text, scoring="r2", X=X, y=y)
result_string = cross_validate(model_string, scoring="r2", X=X, y=y)
# %%
print(f"Base: {result_base['test_score'].mean():.3f}")
print(f"Text encoder: {result_text['test_score'].mean():.3f}")
print(f"String encoder: {result_string['test_score'].mean():.3f}")
# %% With expressions
data = skrub.var("data", df)
X = data.drop("Price").skb.mark_as_X()
y = np.log(data["Price"] + 1e-8).skb.mark_as_y()

# %% Scaling features
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# %%
high_cardinality = skrub.choose_from(
    {"string": StringEncoder(), # "label": LabelEncoder()
     }, name="high_cardinality"
)


low_cardinality = skrub.choose_from(
    {"onehot": OneHotEncoder(sparse_output=False, handle_unknown="ignore"), #"label": LabelEncoder()
     },
    name="low_cardinality",
)

tv = TableVectorizer(high_cardinality=high_cardinality, low_cardinality=low_cardinality)

X = X.skb.apply(tv).skb.apply(StandardScaler())
# %%

transformed = X.skb.apply(StandardScaler())
model = skrub.choose_from(
    {
        # "lr": LinearRegression(),
        "ridge": Ridge(max_iter=500, alpha=0.001, solver="auto"),
        "dt": DecisionTreeRegressor(
            max_depth=skrub.choose_int(5, 20, n_steps=5, name="dt_max_depth"),
            random_state=42,
            min_samples_split=skrub.choose_int(
                2, 20, n_steps=5, name="dt_min_samples_split"
            ),
            min_samples_leaf=skrub.choose_int(
                1, 10, n_steps=5, name="dt_min_samples_leaf"
            ),
        ),
        "rf": RandomForestRegressor(
            max_depth=skrub.choose_int(5, 20, n_steps=5, name="rf_max_depth"),

            random_state=42,
            min_samples_split=skrub.choose_int(
                2, 20, n_steps=5, name="rf_min_samples_split"
            ),
            min_samples_leaf=skrub.choose_int(
                1, 10, n_steps=5, name="rf_min_samples_leaf"
            ),
        ),
        "svr": SVR(
            kernel=skrub.choose_from(["linear", "poly", "rbf"], name="svr_kernel"),
            C=skrub.choose_float(0.001, 0.1, log=True, n_steps=3, name="svr_C"),
            gamma=skrub.choose_from(["scale", "auto"], name="svr_gamma") 
        ),
        "hgb": HistGradientBoostingRegressor(
            learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="hgb_lr"),
        ),
        "xgboost": XGBRegressor(
            learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="xgb_lr"),
            n_estimators=skrub.choose_int(100, 250, log=True, n_steps=5, name="xgb_estimators"),
            max_depth=skrub.choose_int(100, 250, log=True, n_steps=5, name="xgb_max_depth")
        ),
    },
    name="model",
)

# %%
predictions = transformed.skb.apply(model, y=y)
# %%
search  = predictions.skb.get_randomized_search(
    n_jobs=-1, fitted=True, n_iter=64, random_state=42, cv=10
)
# %%
search.best_score_
# %%
search.plot_results()
# %%
