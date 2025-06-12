# %%
import polars as pl
from skrub import TableReport

# %%
df = pl.read_csv("Airbnb_Open_Data.csv")
# %%
TableReport(df)
# %%
df_sample = df.sample(1000)
# df_sample.select("lat", "long").to_numpy()
# %%
from mpl_toolkits.basemap import Basemap

# %%
# m = Basemap()
# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
lons = df_sample["long"].to_numpy()
lats = df_sample["lat"].to_numpy()
lat_center = lats.mean()
lon_center = lons.mean()
# %%
fig, ax = plt.subplots()
sns.scatterplot(
    data=df_sample.to_pandas(), y="lat", x="long", hue="neighbourhood group", ax=ax
)
ax.scatter(y=lat_center, x=lon_center, color="black")
# %%

import numpy as np
import matplotlib.pyplot as plt

margin = 0.1

df = df.drop_nulls(subset=["lat", "long"])

lons = df["long"].to_numpy()
lats = df["lat"].to_numpy()
lat_center = lats.mean()
lon_center = lons.mean()

data_width = abs(np.max(lons) - np.min(lons))
data_height = abs(np.max(lats) - np.min(lats))

lllat = np.min(lats) - margin * data_height
lllon = np.min(lons) - margin * data_height
urlat = np.max(lats) + margin * data_height
urlon = np.max(lons) + margin * data_height


fig, ax = plt.subplots()

m = Basemap(
    projection="cass",
    llcrnrlat=lllat,
    urcrnrlat=urlat,
    llcrnrlon=lllon,
    urcrnrlon=urlon,
    lat_ts=20,
    resolution="h",
    lon_0=lon_center,
    lat_0=lat_center,
    ax=ax,
)

m.drawcoastlines()
m.fillcontinents()
m.drawmeridians([lon_center], labels=[0, 0, 0, 1])
m.drawparallels([lat_center], labels=[1, 0, 0, 0])


# find unique neighborhood groups
unique_groups = df["neighbourhood group"].unique().to_numpy()

# define a colormap with 5 colors
colors = sns.color_palette("husl", len(unique_groups))
# create a dictionary that maps the group to the color
color_dict = dict(zip(df["neighbourhood group"].unique(), colors))

for gidx, group in df.group_by("neighbourhood group"):
    color = color_dict[gidx[0]]
    lons = group["long"].to_numpy()
    lats = group["lat"].to_numpy()
    m.scatter(
        lons, lats, latlon=True, zorder=5, ax=ax, s=0.5, label=gidx[0], color=color
    )


# fig = plt.figure(figsize=(10, 6))

# %%
