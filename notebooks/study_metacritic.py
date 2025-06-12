# %%
import polars as pl
import pandas as pd
from skrub import TableVectorizer, TableReport

# %%
df = pl.read_csv("metacritic.csv", ignore_errors=True)
# %%
# Many columns include a very large fraction of nulls and are likely to be
# uninformative. I will run the TableVectorizer as a preprocessor to clean up
# the table."
tv = TableVectorizer(
    drop_null_fraction=0.4,
    low_cardinality="passthrough",
    high_cardinality="passthrough",
    datetime="passthrough",
    numeric="passthrough",
)

df = tv.fit_transform(df)
# %%
df.columns
# %%
# This reduced the number of columns to a more manageable amount.
TableReport(df)
# %%
# Now, the Associations tab in the table report shows various redundant columns.
# Since URLs are not very informative, and are likely to cause issues with the
# categorical encoders, I will be removing them.
import polars.selectors as cs

# %%
df = df.drop(cs.matches("[Uu]rl"))

# %%
TableReport(df)
# %% For simplicity, we will trim down the columns even further. This can be done
# directly from the TableReport, by selecting the columns and copying the list 
# with their name.  
cols_to_consider = [
    "title",
    "genres/0",
    "metascore",
    "publisherName",
    "releaseDate",
    "section",
    "summary",
    "userscore",
    "platforms/0",
    "platforms/1",
]
df = df[cols_to_consider]
# %% Now that we have a cleaned up table, we can apply some more preprocessing and 
# prepare it for a ML task. In this case, we will try to predict the *metascore*
# based on all features. Note that the *userscore* is a good, but not perfect
# indication of the *metascore*.
#%% First, we will remove null values from the target column, and then scale
# both scores in the [0, 10] range.
df = df.drop_nulls("metascore").with_columns(metascore=pl.col("metascore")/10)

#%%
# We can observe this easly by plotting 
import matplotlib.pyplot as plt
import seaborn as sns
df_scores = df.select("userscore", "metascore").unpivot()
sns.jointplot(data=df.to_pandas(), x="userscore", y="metascore", )
sns.histplot(data=df_scores.to_pandas(), x="value", hue="variable", multiple="dodge")
# %%
# Now that we have a clean set of features to work with, we can encode them using
# the TableVectorizer. We will test two different encoders for categorical values,
# the StringEncoder and the TextEncoder. The first is faster, while the second 
# should be able to leverage the information provided by the pre-trained language
# models. 
# Additionally, we simplify the DatetimeEncoder to keep only the date. 


from skrub import TextEncoder, StringEncoder, TableVectorizer, DatetimeEncoder, MinHashEncoder, GapEncoder

de = DatetimeEncoder(resolution="day")

tv_text = TableVectorizer(high_cardinality=TextEncoder(), datetime=de)
tv_string = TableVectorizer(high_cardinality=StringEncoder(), datetime=de)
tv_gap = TableVectorizer(high_cardinality=GapEncoder(), datetime=de)
tv_minhash = TableVectorizer(high_cardinality=MinHashEncoder(), datetime=de)
# %%
# Then, we can build a scikit-learn pipeline to train a model on the encoded features.
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

model_text = make_pipeline(tv_text, HistGradientBoostingRegressor())
model_string = make_pipeline(tv_string, HistGradientBoostingRegressor())
model_gap = make_pipeline(tv_gap, HistGradientBoostingRegressor())
model_minhash= make_pipeline(tv_minhash, HistGradientBoostingRegressor())
# %%
from sklearn.model_selection import cross_validate
X = df.drop("metascore")
y = df["metascore"]
# results_text = cross_validate(model_text, X, y)
# results_string = cross_validate(model_string, X, y, scoring="r2")
results_gap = cross_validate(model_gap, X, y, scoring="r2")
results_minhash = cross_validate(model_minhash, X, y, scoring="r2")
# %%
# print(results_text)
# print(results_string)
print(results_gap)
print(results_minhash)
# %%
# This is the result of running the code with model_text 
# {'fit_time': array([461.05230474, 448.31556892, 462.2647388 , 468.39861584,
#        424.94758153]), 'score_time': array([117.39304781, 110.9208293 , 114.84054351, 114.45499659,
#        115.76044393]), 'test_score': array([-13.15278203, -20.83451005, -15.22253019,  -9.26460251,
#         -5.1446831 ])}
# This is the result with model_string 
# {'fit_time': array([11.7989018 , 10.84904361, 10.76339722, 11.66449046, 10.77636671]),
#  'score_time': array([1.277668  , 1.15200782, 1.16299105, 1.22888851, 1.25985575]),
#  'test_score': array([-13.22319528, -20.84162851, -15.49173657,  -9.75528685,
#          -5.19542313])}
# Both have terrible performance and are very slow
# This is with model_gap
# {'fit_time': array([121.50642371, 120.82774639, 114.6144824 , 110.14247155,
#        124.071913  ]), 'score_time': array([49.98525858, 50.99296737, 46.30264091, 48.95707393, 50.04634643]), 'test_score': array([-13.7871012 , -21.5246784 , -15.63481294,  -9.77225885,
#         -5.24960576])}
# And this is with model_minhash
# {'fit_time': array([8.99450445, 9.64288497, 9.074435  , 8.76345611, 8.87461066]), 'score_time': array([2.11047769, 1.95113134, 2.15375829, 1.93185401, 1.89055777]), 'test_score': array([-13.56014865, -20.81725396, -15.37740339,  -9.67947036,
#         -5.27652256])}


