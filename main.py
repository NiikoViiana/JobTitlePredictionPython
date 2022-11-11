# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import gc
from tqdm.notebook import tqdm
from mpl_toolkits.basemap import Basemap

warnings.filterwarnings('ignore')

data = pd.read_csv("Dataset/jobss.csv")
print("\n Quick view of the data: \n")
# data.head()
# data.describe()
# data.info()

def mapWorld():
    m = Basemap(projection='mill', llcrnrlat=-60, urcrnrlat=90, \
                llcrnrlon=-180, urcrnrlon=180, resolution='c')
    # m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(-90, 91., 30.))
    m.drawmeridians(np.arange(-90, 90., 60.))

    m.drawmapboundary(fill_color='#FFFFFF')
    lat = data['Latitude'].values
    lon = data['Longitude'].values
    a_1 = data['sal'].values
    a_2 = data['Counts'].values
    m.scatter(lon, lat, latlon=True, c=100 * a_1, s=20 * a_2, linewidth=0, edgecolors='black', cmap='hot', alpha=1)

    m.fillcontinents(color='#072B57', lake_color='#FFFFFF', alpha=0.4)
    cbar = m.colorbar()
    cbar.set_label('Location counts and sal')
    # plt.clim(20000, 100000)
    plt.title("Location counts and sal", fontsize=30)
    plt.show()


sns.set(style="white", font_scale=1.5)
plt.figure(figsize=(30, 30))

data["Counts"] = data["Location"].map(data["Location"].value_counts())
mapWorld()