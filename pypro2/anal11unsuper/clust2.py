# iris dataset으로 군집분석

import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
plt.rc('font', family='malgun gothic')

iris = load_iris()
ir_df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(ir_df.head(2))
print(ir_df.loc[0:4, ['sepal length (cm)', 'sepal width (cm)']])
# dist_vec = pdist(ir_df.loc[0:4, ['sepal length (cm)', 'sepal width (cm)']], metric='euclidean') 
dist_vec = pdist(ir_df.loc[:, ['sepal length (cm)', 'sepal width (cm)']], metric='euclidean')  # 전체 행 모두 참여
print('dist_vec:', dist_vec)
row_dist = pd.DataFrame(squareform(dist_vec))
print(row_dist)

row_cluster = linkage(dist_vec, method='complete')
# print('row_cluster:', row_cluster)
df = pd.DataFrame(row_cluster, columns=['군집1','군집2','거리','멤버수'])
print(df)

row_dend = dendrogram(row_cluster)
plt.ylabel('유클리드 거리')
plt.show()