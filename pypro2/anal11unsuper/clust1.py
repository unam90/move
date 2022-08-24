# 군집분석(Clustering Analysis)
# 계층적 군집분석

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')

np.random.seed(123)

var=['x', 'y']
labels = ['점0','점1','점2','점3','점4']

x = np.random.random_sample([5, 2]) * 10  # 5행 2열짜리
df = pd.DataFrame(x, columns=var, index=labels)
print(df)

plt.scatter(x[:, 0], x[:, 1], s=50, c='blue', marker='o')
plt.grid()
plt.show()

from scipy.spatial.distance import pdist, squareform

dist_vec = pdist(df, metric='euclidean')
print('거리:', dist_vec)

row_dist = pd.DataFrame(squareform(dist_vec), columns=labels, index=labels)  # 표로 만들어서 보여주기 
print(row_dist)

from scipy.cluster.hierarchy import linkage  # 응집형 계층적 군집화 가능
row_clusters = linkage(dist_vec, method='average')  # method에는 ward, centroid 등이 있다.
# print(row_clusters)
df = pd.DataFrame(row_clusters, columns=['군집1','군집2','거리','멤버수'])
print(df)

# linkage의 결과를 계통도(dendrogram)으로 시각화 
from scipy.cluster.hierarchy import dendrogram
dend = dendrogram(row_clusters, labels = labels)
plt.ylabel('유클리드 거리')
plt.show()














