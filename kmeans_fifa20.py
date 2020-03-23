
#%%
import pandas as pd

data = pd.read_csv('fifa20.csv')
data.head()

#%%
dt = data[['overall','age']]
dt.tail()

# %%
#cek ada data yg NaN, kalau ada, drop
dt.dropna()
dt

# %%
#plot untuk lihat data spreadnya
import matplotlib.pyplot as plt

plt.scatter(dt['overall'],dt['age'],color='red')
plt.show()

#%%
ready = dt.values.tolist()
print(ready)

# %%
#proses kmeans nya disini

from sklearn.cluster import KMeans 
km = KMeans(n_clusters = 3)
km.fit(ready)

print(km.labels_)
#%%
#hasil akhir clustering

plt.scatter(dt['overall'],dt['age'],c=km.labels_,cmap='rainbow')
plt.scatter(km.cluster_centers_[:,0] ,km.cluster_centers_[:,1], color='black')
plt.show()


# %%
