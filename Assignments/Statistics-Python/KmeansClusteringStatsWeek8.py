import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_excel('C:/Users/ldmag/Downloads/HBAT.xls')
selected = dataset[['x6','x8','x12','x15','x18']]

'''Requirements:
    - Cluster means
    - Cluster standard deviation
    - Distance between cluster centroids
'''
# finding best number of clusters with elbow method:
error = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i).fit(selected)
    kmeans.fit(selected)
    error.append(kmeans.inertia_)
    
plt.plot(range(1,11), error)
plt.title('Elbow')
plt.xlabel('# of clusters')
plt.ylabel('error')
plt.show()

# clusters = 3
kmeans3 = KMeans(n_clusters = 3, init = 'k-means++')
pred3 = kmeans3.fit_predict(selected)
print(pred3)
kmeans3.cluster_centers_

# clusters = 4
kmeans4 = KMeans(n_clusters = 4, init = 'k-means++')
pred4 = kmeans4.fit_predict(selected)
kmeans4.cluster_centers_

# clusters = 5
kmeans5 = KMeans(n_clusters = 5, init = 'k-means++')
pred5 = kmeans5.fit_predict(selected)
kmeans5.cluster_centers_

# correlation
c = selected.corr()

# visualization
array = selected.to_numpy()
plt.scatter(array[:,0], array[:,1], c= pred3, cmap = 'rainbow')
plt.scatter(array[:,0], array[:,1], c= pred4, cmap = 'rainbow')
plt.scatter(array[:,0], array[:,1], c= pred5, cmap = 'rainbow')

# elbow with yellow brick
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1, 9))
visualizer.fit(array)
visualizer.show()